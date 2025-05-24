import torch
from torch.utils.data import DataLoader, Dataset
from config import TrainCfg
import numpy as np
import torchvision
from torchvision import transforms


class MyCIFAR10:
    def __init__(self, config: TrainCfg):
        self.cfg = config
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # 设置均值和方差
        ])
        self._full_train_set = torchvision.datasets.CIFAR10(
            root=self.cfg.DataRoot,
            train=True,
            transform=transform)
        self._full_val_set = torchvision.datasets.CIFAR10(
            root=self.cfg.DataRoot,
            train=False,
            transform=transform)
        print("训练集数据数量：{}".format(len(self._full_train_set)))
        print("验证集数据数量：{}".format(len(self._full_val_set)))
        self.train_set = None
        self.val_set = None
        self.train_loader = None
        self.val_loader = None

    def load_data(self):
        self.train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.cfg.BatchSize,
            shuffle=True,
            drop_last=True)
        self.val_loader = DataLoader(
            dataset=self.val_set,
            batch_size=self.cfg.BatchSize,
            shuffle=True,
            drop_last=True)

    def stratified_sampling(self):
        """
        按类别分层抽样
        """
        np.random.seed(self.cfg.seed)
        class_indices = {}
        # 训练集分组
        for idx, label in enumerate(self._full_train_set.targets):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        # 采样
        subset_indices = []
        for label in class_indices:
            indices = class_indices[label]
            num_samples = int(len(indices) * self.cfg.ratio)
            subset_indices.extend(np.random.choice(indices, num_samples, replace=False))
        # 构建训练集
        subset_indices = np.array(subset_indices)
        self.train_set = torch.utils.data.Subset(self._full_train_set, subset_indices)

        class_indices.clear()
        subset_indices = []

        # 验证集分组
        for idx, label in enumerate(self._full_val_set.targets):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        # 采样
        val_subset_indices = []
        for label in class_indices:
            indices = class_indices[label]
            num_samples = int(len(indices) * self.cfg.ratio)
            val_subset_indices.extend(np.random.choice(indices, num_samples, replace=False))
        # 构建验证集
        val_subset_indices = np.array(val_subset_indices)
        self.val_set = torch.utils.data.Subset(self._full_val_set, val_subset_indices)
        print("训练集使用数据数量：{}".format(len(self.train_set)))
        print("验证集使用数据数量：{}".format(len(self.val_set)))
        return subset_indices


if __name__ == "__main__":
    cfg = TrainCfg()
    my_cifar10 = MyCIFAR10(cfg)
    my_cifar10.stratified_sampling()
    my_cifar10.load_data()
    for img, target in my_cifar10.train_loader:
        print(img.shape)
        break
