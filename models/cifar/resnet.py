import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.rate = 1.0
        self.inplanes = 16
        super(CifarResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(16 * self.rate), layers[0])
        self.layer2 = self._make_layer(block, int(32 * self.rate), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(64 * self.rate), layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(int(64 * block.expansion * self.rate), num_classes)
        self.intermediate = []
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        out = self.relu(x)
        # x = self.maxpool(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        # import pdb
        # pdb.set_trace()
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet56(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, [9, 9, 9], num_classes)
    return model


def resnet110(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, [18, 18, 18], num_classes)
    return model

if __name__ == '__main__':
    import torch
    model = resnet110(num_classes=10)
    conv_count = 1
    l1 = []
    l2 = []
    l3 = []
    skip = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if '.conv1' in name:
                l1.append(conv_count)
            elif '.conv2' in name:
                l2.append(conv_count)
            elif '.conv3' in name:
                l3.append(conv_count)
            else:
                skip.append(conv_count)
            conv_count += 1
    print(l1)
    print(l2)
    print(l3)
    print(skip)
