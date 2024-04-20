import torch
import torch.nn.functional as F
from lisa import Lisa
from adam import Adam

from timm.utils import accuracy, AverageMeter

from AdaBelief import AdaBelief


def test(model, device, test_loader, writer, i):
    test_loss = 0
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            test_loss += F.cross_entropy(pred, target, reduction='sum').item()  # sum up batch loss
            top1, top5 = accuracy(pred, target, topk=(1,5))
            acc1_meter.update(top1.item(), target.size(0))
            acc5_meter.update(top5.item(), target.size(0))

    test_loss /= len(test_loader.dataset)
    writer.add_scalar("Loss/test", test_loss, i)
    writer.add_scalar("Accuracy-top1/test", acc1_meter.avg, i)
    writer.add_scalar("Accuracy-top5/test", acc5_meter.avg, i)

    print(f"\nTest set: Average loss: {test_loss:.4e}, Accuracy-top1: {acc1_meter.avg:.2f}%, Accuracy-top5: {acc5_meter.avg:.2f}%\n")


def train_lisa(args, model, device, train_dataset, test_loader, writer):
    optimizer = Lisa(
        model,
        train_dataset,
        args.alpha,
        weight_decay=args.weight_decay,
        N0=args.batch_size,
        vm=args.optim == "lisa-vm",
        betas=(args.beta1, args.beta2, args.beta3),
        steps=args.steps,
        eps_k_fact=args.eps_k_fact,
        ls_ci=args.ls_ci,
        gamma1=args.gamma1,
        writer=writer
    )
    
    model.train()

    def closure(sample):
        data, target = sample
        output = model(data)
        return F.cross_entropy(output, target, reduction='none')
    
    # def closure(params, buffers, data, target):
    #     if data.ndim == 3:
    #         data = data.unsqueeze(0)
    #         target = target.unsqueeze(0)
    #     output = torch.func.functional_call(model, (params, buffers), (data,))
    #     return F.cross_entropy(output, target, reduction='mean')

    for i in range(args.steps):
        loss = optimizer.step(closure)
        writer.add_scalar("Loss/train", loss.item(), i)
        if i % args.log_interval == 0:
            print(f"Train Step: {i:5d} alpha={optimizer.alpha:.3e}  N_k={optimizer.Nk:3d}\tLoss: {loss.item():.4e}")
        if i % args.test_interval == 0:
            test(model, device, test_loader, writer, i)


def train_std(args, model, device, train_loader, test_loader, writer):
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.alpha,
            momentum=args.acc,
            weight_decay=args.weight_decay
        )
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.alpha,
            betas=(args.acc, args.beta2),
            weight_decay=args.weight_decay
        )
    elif args.optim == "adabelief":
        optimizer = AdaBelief(
            model.parameters(),
            lr=args.alpha,
            eps=1e-16,
            acc=args.acc,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            writer=writer
        )
    
    model.train()

    liter = iter(train_loader)

    def sample_single(liter):
        try:
            sample = next(liter)
        except StopIteration:
            liter = iter(train_loader)
            sample = next(liter)
        return sample, liter

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                eta_min=1e-6,
                T_max=args.steps,
            )
    for i in range(args.steps):
        (data, target), liter = sample_single(liter)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        writer.add_scalar("Loss/train", loss.item(), i)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % args.log_interval == 0:
            print(f"Train Step: {i:5d} alpha={scheduler.get_last_lr()[0]:.3e} N_k={data.shape[0]:3d}\tLoss: {loss.item():.4e}")
        if i % args.test_interval == 0:
            model.eval()
            test(model, device, test_loader, writer, i)
            model.train()
