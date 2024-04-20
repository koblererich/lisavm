from torch import nn
from torchvision.models import resnet18, squeezenet1_1, shufflenet_v2_x0_5

from models_architectures import *

from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from torchvision.models import resnet


class MnistNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        x = 16 if num_channels == 1 else 25
        self.fc1 = nn.Linear(x*32, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc1(x)



class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


def get(type: str, num_channels: int = 3, num_classes: int = 200, use_bn: bool = True):
    model = None
    ## Models for cifar10
    if type == "simple":
        return MnistNet(num_channels, num_classes)
    elif type == "vgg":
         model = VGG('VGG11', num_classes=num_classes, use_bn=False)
    elif type == "resnet188":
         model = ResNet18(num_classes, use_bn=False)
    elif type == "preactresnet18":
         model = PreActResNet18(num_classes)
    elif type == "googlenet":
         model = GoogLeNet()
    elif type == "densenet121":
         model = DenseNet121()
    elif type == "resnext29_2x64d":
         model = ResNeXt29_2x64d()
    elif type == "mobilenet":
         model = MobileNet(num_classes=num_classes)
    elif type == "mobilenetv2":
         model = MobileNetV2(num_classes=num_classes)
    elif type == "dpn92":
         model = DPN92()
    elif type == "shufflenetg2":
         model = ShuffleNetG2()
    elif type == "senet18":
         model = SENet18()
    elif type == "shufflenetv2":
         model = ShuffleNetV2(1)
    elif type == "efficientnetb0":
         model = EfficientNetB0()
    elif type == "regnetx_200mf":
         model = RegNetX_200MF()
    elif type == "simpledla":
         model = SimpleDLA()
         
    # models for tiny imagenet
    elif type == "resnet18":
        # no pretrained model
        model = resnet18(weights=None)
        # replace the fully connected head to fit to the tiny imagenet classes
        model.fc.out_features = num_classes
        # replace all inplace operations
        model.relu = nn.ReLU()
        def replace_layers(model):
            for n, module in model.named_children():
                if len(list(module.children())) > 0:
                    ## compound module, go inside it
                    replace_layers(module)
                    
                if isinstance(module, resnet.BasicBlock):
                    ## simple module
                    new = BasicBlock(
                        inplanes=module.conv1.in_channels,
                        planes=module.conv1.out_channels,
                        stride=module.stride,
                        downsample=module.downsample
                    )
                    setattr(model, n, new)

        replace_layers(model)
                
    elif type == "shufflenet":
        # no pretrained model
        model = shufflenet_v2_x0_5(weights=None)
        # replace the fully connected head to fit to the tiny imagenet classes
        model.fc.out_features = num_classes
        # model.fc = nn.Linear(model.fc.in_features, classes)
    elif type == "squeezenet":
        # no pretrained model
        model = squeezenet1_1(weights=None)
        model.features[0].kernel_size = 3
        model.features[0].stride = 1
        model.features[2] = nn.Identity()
        model.classifier[0] = nn.Identity()
        # replace the fully connected head to fit to the tiny imagenet classes
        model.classifier[1].out_features = num_classes
        # model.classifier[1] = nn.Conv2d(512, classes, kernel_size=1)
        def replace_layers(model):
            for n, module in model.named_children():
                if len(list(module.children())) > 0:
                    ## compound module, go inside it
                    replace_layers(module)
                    
                if isinstance(module, nn.ReLU):
                    ## simple module
                    setattr(model, n, nn.ReLU())

        replace_layers(model)
    if model:
        return model
    else:
        raise RuntimeError(f"Model '{type}' not supported!")
