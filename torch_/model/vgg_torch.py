import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
from torchvision.models import vgg16 as vgg16_official, vgg11 as vgg11_official


class VGGBase(nn.Module):
    """VGG系列的基类"""

    def __init__(self, features, num_classes=10, pretrained=False, vgg_version='vgg16'):
        super(VGGBase, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.vgg_version = vgg_version
        if pretrained:
            self.load_pretrained_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16(VGGBase):
    """VGG16模型（继承自VGGBase）"""

    def __init__(self, num_classes=10, pretrained=False):
        features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        super(VGG16, self).__init__(
            features=features,
            num_classes=num_classes,
            pretrained=pretrained,
            vgg_version='vgg16'
        )


# 扩展示例：VGG11模型
class VGG11(VGGBase):
    def __init__(self, num_classes=10, pretrained=False):
        features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        super(VGG11, self).__init__(
            features=features,
            num_classes=num_classes,
            pretrained=pretrained,
            vgg_version='vgg11'
        )


def load_pretrained_weights(weight_path):
    """加载PyTorch官方对应版本的预训练参数"""
    pretrained_dict = torch.load(weight_path)
    new_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if 'classifier' in k:
            new_k = k.replace('classifier', 'fc')
        else:
            new_k = k
        new_dict[new_k] = v
    return new_dict


class CIFAR10Quick(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Quick, self).__init__()
        self.features = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256x4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CIFAR10Simple(nn.Module):
    """模型结构图"""

    def __init__(self):
        super(CIFAR10Simple, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input_img):
        """输入图像，前向传播"""
        result = self.model(input_img)
        return result


# 加载适配后的参数

# 使用示例
if __name__ == "__main__":
    torch.manual_seed(42)
    model1= CIFAR10Simple()
    model2 = CIFAR10Quick()
    test = torch.ones(1, 3, 32, 32)
    output1 = model1(test)
    output2 = model2(test)
    print("student:", output1.data)
    print("teacher:", output2.data)

