import jittor as jt
from jittor import nn

class CIFAR10Quick(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Quick, self).__init__()
        self.model = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Pool(kernel_size=2, stride=2),  # 64x16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Pool(kernel_size=2, stride=2),  # 128x8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Pool(kernel_size=2, stride=2),  # 256x4x4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )


    def execute(self, x):
        x = self.model(x)
        return x


class CIFAR10Simple(nn.Module):
    def __init__(self):
        super(CIFAR10Simple, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.Pool(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, 5, 1, 2),
            nn.Pool(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, 5, 1, 2),
            nn.Pool(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )


    def execute(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    jt.misc.set_global_seed(42, False)
    model1 = CIFAR10Simple()
    model2 = CIFAR10Quick()
    test = jt.ones((1, 3, 32, 32))
    output1 = model1(test)
    output2 = model2(test)
    print("student:",model1.parameters())
    print("student",output1.data)
    print("teacher:",output2.data)
