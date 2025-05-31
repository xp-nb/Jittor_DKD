import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """residual block"""

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """输入通道数，输出通道数，卷积步长，下采样模块"""
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        """forward function"""
        # indentity
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        # residual
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        # add
        feature = identity + residual
        out = F.relu(feature)
        return out


class ResNet(nn.Module):
    def __init__(self, depth, channels, num_classes=10):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
        n = (depth - 2) // 6
        block = ResidualBlock

        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, channels[0], channels[1], n)
        self.layer2 = self._make_layer(block, channels[1], channels[2], n, stride=2)
        self.layer3 = self._make_layer(block, channels[2], channels[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(channels[3], num_classes)
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, block_num, stride=1, is_downsample=True):
        """模块种类，输入通道数，输出通道数，块数量，卷积步长"""
        # 封装下采样模块
        if is_downsample:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False,),
                nn.BatchNorm2d(planes),
            )
        else:
            downsample = None
        # 层数组
        layers = list([])
        # 添加第一个块
        layers.append(block(inplanes, planes, stride, downsample))
        # 添加剩余的块
        for i in range(1, block_num):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x = self.layer1(x)  # 32x32
        f1 = x
        x = self.layer2(x)  # 16x16
        f2 = x
        x = self.layer3(x)  # 8x8
        f3 = x

        x = self.avgpool(x)
        avg = x.reshape(x.size(0), -1)
        out = self.fc(avg)
        return out


def resnet8x4():
    """网络深度，[网络输入，第一层，第二层，第三层]尺寸"""
    return ResNet(8, [32, 64, 128, 256])


def resnet32x4():
    """网络深度，[网络输入，第一层，第二层，第三层]尺寸"""
    return ResNet(32, [32, 64, 128, 256])


if __name__ == '__main__':
    Smodel = resnet8x4()
    Tmodel = resnet32x4()
    print(Smodel, '\n', Tmodel)
