import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from lisa import autograd_hacks
import models



def test_grad1():
    torch.manual_seed(1)

    model = models.get("vgg", num_channels=3, num_classes=1)
    print(model)

    loss_fn = nn.MSELoss(reduction="sum")

    n = 4
    data = torch.rand(n, 3, 28, 28)
    targets = torch.rand(n)

    model.eval()
    autograd_hacks.add_hooks(model)
    output = model(data)
    loss_fn(output, targets).backward(retain_graph=True)
    autograd_hacks.compute_grad1(model, loss_type="sum")
    autograd_hacks.disable_hooks()

    # Compare values against autograd
    losses = torch.stack([loss_fn(output[i:i+1], targets[i:i+1]) for i in range(len(data))])

    for layer in model.modules():
        if not autograd_hacks.is_supported(layer):
            continue
        print(layer)
        for n, param in layer.named_parameters():
            print(torch.abs(param.grad-param.grad1.sum(dim=0)).max())


if __name__ == '__main__':
    test_grad1()
