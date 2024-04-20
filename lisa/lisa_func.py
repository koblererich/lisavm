import math
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from scipy.special import erfinv
import matplotlib.pyplot as plt

class Lisa(Optimizer):
    r"""LISA

        TODO: provide some details
    """

    def __init__(self, model, dataset, alpha_init=1e-0, alpha_max=1., weight_decay=0, betas=(0.9, 0.9, 0.999), steps=1, delta1=1/3, delta2=2/3, 
                 gamma1=100, N0=10, nb=30, eps_k_fact=1., ls_ci=.8, vm=False, writer=None):
        defaults = dict()
        super().__init__(model.parameters(), defaults)

        self.k = 0
        # number of samples
        self._model = model
        self._device = list(model.parameters())[0].device
        self._dataset = dataset 
        # self._loader = DataLoader(self._dataset, batch_size=1, shuffle=True, num_workers=4, prefetch_factor=N0)
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

        self.var_k = 1
        self.gamma1 = gamma1
        self.eps_k_fact = eps_k_fact
        self.gamma2 = math.sqrt(2) * erfinv(2*ls_ci - 1)  # compute confidence interval for non-montone LS
        print(f"gamma2={self.gamma2} ci={ls_ci}", "-"*20)

        self.writer = writer
        self.fig, self.ax = plt.subplots(2,1)

        self._params = {k: v.detach() for k, v in model.named_parameters()}
        self._buffers = {k: v.detach() for k, v in model.named_buffers()}
        self._compute_sample_grad = None

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
        q = n//self.N0
        r = n%self.N0
        if r==0:
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

    def _grads_to_vec(self, grads):
        return torch.concat([p.flatten(1) for p in grads.values()], dim=1)

    @staticmethod
    def _compute_variance(v, D, dim=(0, 1)):
        return (v - v.mean(dim=0, keepdim=True)).square_().mul_(D).sum(dim=dim)

    def step(self, closure):
        if closure is None:
            raise ValueError("Specify closure to compute the loss")

        sigma = math.sqrt(-self.steps**2/math.log(1/self.gamma1)/2)
        eps_k = math.exp(-self.k**2/sigma**2/2)
        self._log("eps_k", eps_k)

        # switch off normalization during gradient computation
        self._model.eval()

        if self._compute_sample_grad is None:
            ft_compute_grad = torch.func.grad_and_value(closure)
            self._compute_sample_grad = torch.func.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
            

        # estimate the sample-wise gradient and its variance
        y = self._param_to_vec()

        f_k = y.new_zeros((0,))
        var_k = torch.inf

        if self.k == 0:
            self.D_k = torch.ones_like(y)

        # sample batch
        samples = []
        nabla_f_k = y.new_zeros((0, y.shape[0]))
        n = 0

        self.zero_grad()

        if self.k == 0:
            self.var_k = 0.

        while True:
            new_samples = self._sample_data(self.Nk - n)
            # tmp = closure(new_samples)
            # tmp.mean().backward(retain_graph=True)
            grads, values = self._compute_sample_grad(self._params, self._buffers, new_samples[0], new_samples[1])
            # accumulate the samples
            samples.append(new_samples)
            # accumulate the loss
            f_k = torch.concat([f_k, values], dim=0)
            # accumulate the gradients
            nabla_f_k = torch.concat([nabla_f_k, self._grads_to_vec(grads)], dim=0)
            var_k = self.alpha * self._compute_variance(nabla_f_k, self.D_k)/(self.Nk*(self.Nk-1))

            s_var_k = self.var_k * self.beta1 + (1 - self.beta1) * var_k
            n = self.Nk

            # check if variance condition is fulfilled
            s_var_k_bc = s_var_k / (1 - self.beta1**(self.k+1))
            if s_var_k_bc <= eps_k * self.gamma1:
                break
            else:
                self.Nk = min(self.N, max(int(self.Nk * s_var_k_bc / (eps_k * self.gamma1)), self.Nk+1))
                self.Nk = min(self.Nk, 160)  # accounting for the limited GPU ressources
                if n == self.Nk:
                    break
        self.var_k = s_var_k

        self._log("var_f_k", f_k.var())
        self._log("var_k", var_k)
        self._log("s_var_k", s_var_k)
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

            bias_correction = 1 - self.beta3 ** (self.k + 1)

            self.D_k = 1 / (self.g_k_exp_var.add_(1e-16).sqrt() / math.sqrt(bias_correction)).add_(1e-16)
        else:
            self.D_k = torch.ones_like(y)
        self._log("min(D_k)", self.D_k.min())
        self._log("max(D_k)", self.D_k.max())

        # if self.k % 500 == 0:
        #     self._log_var(self.A_k, self.D_k)

        for i in range(self.nb):
            # compute the new estimate
            x_bar = y - self.alpha * self.D_k * g_k
            self._vec_to_param(x_bar)

            with torch.no_grad():
                f_new = closure(self._params, self._buffers, samples[0], samples[1]).mean()

            diff = x_bar-y[None, ...]
            condition_k = f_k + (nabla_f_k*diff).sum(dim=1) + diff.square().div(self.D_k).sum()/self.alpha/2
            condition = condition_k.mean().item() + self.gamma2 * condition_k.std().item() / math.sqrt(self.Nk) * eps_k

            # print(f"{i:2d}: f_k={f_k.mean().item():.4e} {c_k=:.4e} f_new={f_new.item():.4e} {condition=:.4e} {self.alpha:.4e}")
            if f_new.item() <= condition:
                self.alpha = min(self.alpha/self.delta2, self.alpha_max)
                break
            else:
                self.alpha *= self.delta1

        # L2 squared proximal operator part of LISA
        if self.weight_decay!=0:
            self._vec_to_param(x_bar/(1+self.alpha*self.D_k*self.weight_decay))

        self._log("alpha", self.alpha)
        self._log("f_new", f_new)

        self.k += 1


        # # FIXME: turn on if batch normalization is used
        # # update batch normalization
        # self._model.train()
        # with torch.no_grad():
        #     loss = closure(samples)
        
        return f_new
