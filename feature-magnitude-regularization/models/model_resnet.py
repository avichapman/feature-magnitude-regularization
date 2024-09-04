from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

from tools.argument_helper import ArgumentHelper

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    ArgumentHelper.check_type(in_planes, int)
    ArgumentHelper.check_type(out_planes, int)
    ArgumentHelper.check_type(stride, int)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalize(nn.Module):
    def __init__(self, power: float = 2):
        super().__init__()
        self.power = ArgumentHelper.make_type(power, float)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()

        ArgumentHelper.check_type(in_planes, int)
        ArgumentHelper.check_type(planes, int)
        ArgumentHelper.check_type(stride, int)

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = ArgumentHelper.check_type_or_none(downsample, nn.Module)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()

        ArgumentHelper.check_type(in_planes, int)
        ArgumentHelper.check_type(planes, int)
        ArgumentHelper.check_type(stride, int)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ArgumentHelper.check_type_or_none(downsample, nn.Module)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 layers: List[int],
                 projector_dim: int = 1000,
                 in_channel: int = 3,
                 width: int = 1):
        self.in_planes = 64

        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(self.base * 8 * block.expansion, projector_dim)
        self.l2norm = Normalize(2)
        self.out_features = self.base * 8 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs an input tensor through the model, returning feature vectors and a convolutional feature map.
        :param x: The input tensor.
        :return: A tuple containing:
        - The output feature vectors
        - A spatial map of the features from the final convolutional layer.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        conv_feature_maps = self.layer4(x4)

        feature_vectors = self.avgpool(conv_feature_maps)
        feature_vectors = feature_vectors.view(feature_vectors.size(0), -1)
        return feature_vectors, conv_feature_maps


def resnet18(pretrained: bool = False, weights_file: str = None, model_dir: str = None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        weights_file (str): If provided, load the weights from this location.
        model_dir (str): If provided, the location where the weights live or should be downloaded to.
    """
    if weights_file is not None:
        assert model_dir is not None, "model_dir must be provided if weights_file was provided."

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        if weights_file is not None:
            model.load_state_dict(model_zoo.load_url(weights_file, model_dir=model_dir))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir))

    return model


def resnet34(pretrained: bool = False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained: bool = False, weights_file: str = None, model_dir: str = None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        weights_file (str): If provided, load the weights from this location.
        model_dir (str): If provided, the location where the weights live or should be downloaded to.
    """
    if weights_file is not None:
        assert model_dir is not None, "model_dir must be provided if weights_file was provided."

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if weights_file is not None:
            model.load_state_dict(model_zoo.load_url(weights_file, model_dir=model_dir))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir=model_dir))
    return model


def resnet101(pretrained: bool = False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained: bool = False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
