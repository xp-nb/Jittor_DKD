import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from collections import defaultdict
from config import TrainCfg
import numpy as np
import torchvision
from torchvision import transforms


class MyCIFAR10:
    def __init__(self,  config: TrainCfg, train=True):
        self.cfg = config
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # 设置均值和方差
            ])
        else:
            transform = transforms.Compose([
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
            drop_last=False)
        self.val_loader = DataLoader(
            dataset=self.val_set,
            batch_size=self.cfg.BatchSize,
            shuffle=True,
            drop_last=False)

    def stratified_sampling(self):
        """
        按类别分层抽样
        """
        torch.manual_seed(42)
        ratio = self.cfg.ratio
        tclass = defaultdict(list)
        for idx in range(len(self._full_train_set)):
            _, label = self._full_train_set[idx]
            tclass[label].append(idx)
        tselected_indices = []
        for label, indices in tclass.items():
            n_selected = int(len(indices) * ratio)
            tselected_indices.extend(indices[:n_selected])
        tselected_indices = sorted(tselected_indices)
        self.train_set = Subset(self._full_train_set, tselected_indices)

        vclass = defaultdict(list)
        for idx in range(len(self._full_val_set)):
            _, label = self._full_val_set[idx]
            vclass[label].append(idx)
        vselected_indices = []
        for label, indices in vclass.items():
            n_selected = int(len(indices) * ratio)
            vselected_indices.extend(indices[:n_selected])
        vselected_indices = sorted(vselected_indices)
        self.val_set = Subset(self._full_val_set, vselected_indices)
        print("训练集数据数量：{}".format(len(self.train_set)))
        print("验证集数据数量：{}".format(len(self.val_set)))





if __name__ == "__main__":
    cfg = TrainCfg()
    my_cifar10 = MyCIFAR10(cfg)
    # my_cifar10.stratified_sampling()
    # my_cifar10.load_data()
    # writer = SummaryWriter("tool")
    # for i, (x, y) in enumerate(my_cifar10.train_set):
    #     writer.add_images('input', x.unsqueeze(0), i)