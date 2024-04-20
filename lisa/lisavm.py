import math
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from scipy.special import erfinv
import matplotlib.pyplot as plt

from . import autograd_hacks

class Lisa(Optimizer):
    """LISA
    """

    def __init__(self, model, dataset, alpha_init=1e-0, alpha_max=1., weight_decay=0, betas=(0.9, 0.9, 0.999), steps=1, delta1=2/3, delta2=2/3, 
                 gamma1=100, N0=10, nb=30, eps_k_fact=1., ls_ci=.8, vm=False, writer=None):
        defaults = dict()
        super().__init__(model.parameters(), defaults)

        self.k = 0
        # number of samples
        self._model = model
        self._device = list(model.parameters())[0].device
        self._dataset = dataset 
        self._loader = DataLoader(self._dataset, batch_size=N0, shuffle=True, num_workers=4)
        self._iter = iter(self._loader)
        self.Nk = N0
        self.N0 = N0
        self.N = len(dataset)

        self.nb = nb
        self.alpha = alpha_init
        self.alpha_max = alpha_max
        self.vm = vm
        self.weight_decay = weight_decay
        self.beta1, self.beta2, self.beta3 = betas
        self.delta1 = delta1
        self.delta2 = delta2
        self.steps = steps

        self.gamma1 = gamma1
        self.eps_k_fact = eps_k_fact
        self.gamma2 = math.sqrt(2) * erfinv(2*ls_ci - 1)  # compute confidence interval for non-montone LS
        print(f"gamma2={self.gamma2} ci={ls_ci}", "-"*20)

        self.writer = writer
        self.fig, self.ax = plt.subplots(2,1)

        # add the autograd hooks to compute sample-wise gradient
        autograd_hacks.add_hooks(model)

    def _log(self, label, value):
        if self.writer:
            self.writer.add_scalar(f"LISA/{label}", value, self.k)

    def _log_var(self, A, D):
        if self.writer:
            self.ax[0].clear()
            self.ax[0].semilogy(A.cpu())
            self.ax[1].clear()
            self.ax[1].semilogy(D.cpu())
            self.writer.add_figure("LISA/variance", self.fig, self.k, close=False)

    def __setstate__(self, state):
        super().__setstate__(state)

    def _sample_single(self):
        try:
            sample = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            sample = next(self._iter)
        return sample

    def _sample_data(self, n):
        q = n // self.N0
        r = n % self.N0
        if r == 0:
            samples = [self._sample_single() for _ in range(q)]
        else:
            samples = [self._sample_single() for _ in range(q+1)]
            samples[-1] = [l[:r,] for l in samples[-1]]
        return tuple(torch.concat(l, dim=0).to(self._device) for l in zip(*samples))

    def _param_to_vec(self):
        return torch.concat([p.data.ravel() for p in self._model.parameters()])
    
    def _vec_to_param(self, v):
        i = 0
        for p in self._model.parameters():
            n = p.data.numel()
            p.data.copy_(v[i:i+n].reshape(p.data.shape))
            i += n

    def _grad1_to_vec(self):
        return torch.concat([p.grad1.reshape(p.grad1.shape[0], -1) for p in self._model.parameters()], dim=1)

    @staticmethod
    def _compute_variance(v, D, dim=(0, 1)):
        return (v - v.mean(dim=0, keepdim=True)).square_().mul_(D).sum(dim=dim)

    def step(self, closure):
        if closure is None:
            raise ValueError("Specify closure to compute the loss")

        sigma = math.sqrt(-self.steps**2/math.log(1/100)/2)
        eps_k = math.exp(-self.k**2/sigma**2/2)
        self._log("eps_k", eps_k)

        # switch off normalization during gradient computation
        self._model.eval()

        # estimate the sample-wise gradient and its variance
        y = self._param_to_vec()

        f_k = y.new_zeros((0,))
        var_k = torch.inf

        if self.k == 0:
            self.D_k = torch.ones_like(y)
            self.var_k_exg_avg = torch.as_tensor(0, dtype=torch.float32, device=y.device)
            self.var_k_exp_var = torch.as_tensor(0, dtype=torch.float32, device=y.device)

        # sample batch
        samples = []
        nabla_f_k = y.new_zeros((0, y.shape[0]))
        n = 0

        self.zero_grad()

        bias2_correction = 1 - self.beta2 ** (self.k + 1)
        bias3_correction = 1 - self.beta3 ** (self.k + 1)
        if self.k == 0:
            var_bound = eps_k * self.gamma1
        else:
            var_bound = min(eps_k * self.gamma1, self.var_k_exg_avg/bias2_correction + 4 * (self.var_k_exp_var/bias3_correction).sqrt())

        while True:
            autograd_hacks.enable_hooks()
            new_samples = self._sample_data(self.Nk - n)
            tmp = closure(new_samples)
            tmp.mean().backward(retain_graph=True)
            # accumulate the samples
            samples.append(new_samples)
            # accumulate the loss
            f_k = torch.concat([f_k, tmp], dim=0)
            autograd_hacks.compute_grad1(self._model)
            autograd_hacks.disable_hooks()
            # accumulate the gradients
            nabla_f_k = torch.concat([nabla_f_k, self._grad1_to_vec()], dim=0)
            autograd_hacks.clear_backprops(self._model)
            var_k = self._compute_variance(nabla_f_k, torch.ones_like(self.D_k))/(self.Nk*(self.Nk-1))

            n = self.Nk

            # check if variance condition is fulfilled
            if var_k <= var_bound:
                break
            else:
                self.Nk = min(self.N, max(int(self.Nk * var_k / var_bound), self.Nk+1))
                self.Nk = min(self.Nk, 150)  # accounting for the limited GPU ressources
                if n == self.Nk:
                    break

        self.var_k_exg_avg.mul_(self.beta2).add_(var_k, alpha=1 - self.beta2)
        var_residual = var_k - self.var_k_exg_avg
        self.var_k_exp_var.mul_(self.beta3).addcmul_(var_residual, var_residual, value=1 - self.beta3)

        self._log("var_bound", var_bound)
        self._log("var_f_k", f_k.var())
        self._log("var_k", var_k)
        self._log("N_k", self.Nk)

        # combine samples to minibatch
        samples = tuple(torch.concat(l, dim=0) for l in zip(*samples))

        # compute the gradient 
        g_k = nabla_f_k.mean(dim=0)

        # compute the preconditioning
        if self.vm:
            if self.k == 0:
                self.g_k_exg_avg = torch.zeros_like(y)
                self.g_k_exp_var = torch.zeros_like(y)
            self.g_k_exg_avg.mul_(self.beta2).add_(g_k, alpha=1 - self.beta2)
            grad_residual = g_k - self.g_k_exg_avg
            self.g_k_exp_var.mul_(self.beta3).addcmul_(grad_residual, grad_residual, value=1 - self.beta3)

            self.D_k = 1 / (self.g_k_exp_var.add_(1e-16).sqrt() / math.sqrt(bias3_correction)).add_(1e-16)
        else:
            self.D_k = torch.ones_like(y)
        self._log("min(D_k)", self.D_k.min())
        self._log("max(D_k)", self.D_k.max())

        for i in range(self.nb):
            # compute the new estimate
            x_bar = y - self.alpha * self.D_k * g_k

            # L2 squared proximal operator part of LISA
            if self.weight_decay != 0:
                self._vec_to_param(x_bar/(1+self.alpha*self.D_k*self.weight_decay))
            else:
                self._vec_to_param(x_bar)

            with torch.no_grad():
                f_new = closure(samples).mean()

            diff = x_bar-y[None, ...]
            condition = f_k.mean().item() + self.gamma2 * eps_k * f_k.std().item() / math.sqrt(self.Nk) + (g_k * diff).sum() + diff.square().div(self.D_k).sum()/self.alpha/2

            # print(f"{i:2d}: f_k={f_k.mean().item():.4e} {c_k=:.4e} f_new={f_new.item():.4e} {condition=:.4e} {self.alpha:.4e}")
            if f_new.item() <= condition + 1e-6:
                self.alpha = min(self.alpha/self.delta2, self.alpha_max)
                break
            else:
                self.alpha *= self.delta1

        self._log("ls-n-calls", i+1)

        # decrease the mini-batch size
        self.Nk = max(int(self.Nk * self.delta2), self.N0) 

        self._log("alpha", self.alpha)
        self._log("f_new", f_new)

        self.k += 1
        
        return f_new
