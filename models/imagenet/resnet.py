import torch
import torchvision
import torch.nn as nn

def resnet18(num_classes=1000):
    """Constructs a ResNet-50 model for ImageNet (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = torchvision.models.resnet18(num_classes=num_classes)
    return model

def resnet50(num_classes=1000):
    """Constructs a ResNet-50 model for ImageNet (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = torchvision.models.resnet50(num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = resnet18()
    print(model)
    # count = 1
    # l = []
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         print(name, count)
    #         if '.conv1' in name:
    #             l.append(count)
    #         count += 1
    # print(l)
