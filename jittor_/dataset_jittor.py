import jittor
from jittor.dataset.cifar import CIFAR10
from jittor.dataset import Dataset
from jittor.dataset import DataLoader
import jittor.transform as transforms
from collections import defaultdict

class Subset(Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class MyCIFAR10:
    def __init__(self, ratio=0.1, train=True):
        self.ratio = ratio
        self.DataRoot = "/home/xp/jt/data"
        jittor.misc.set_global_seed(42, False)
        if train:
            transform = transforms.Compose([
                transforms.Resize(40),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # 设置均值和方差
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # 设置均值和方差
            ])

        self._full_train_set = CIFAR10(
            root=self.DataRoot,
            train=True,
            transform=transform,
            target_transform=None,
            download=False
        )
        self._full_val_set = CIFAR10(
            root=self.DataRoot,
            train=False,
            transform=transform,
            target_transform=None,
            download=False
        )
        print("训练集数据数量：{}".format(len(self._full_train_set)))
        print("验证集数据数量：{}".format(len(self._full_val_set)))
        self.train_set = None
        self.val_set = None
        self.train_loader = None
        self.val_loader = None

    def load_data(self):
        self.train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=128,
            shuffle=True,
            drop_last=False)
        self.val_loader = DataLoader(
            dataset=self.val_set,
            batch_size=128,
            shuffle=True,
            drop_last=False)

    def stratified_sampling(self):
        """
        按类别分层抽样
        """
        ratio = self.ratio
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
    my_cifar10 = MyCIFAR10()
    my_cifar10.stratified_sampling()
    my_cifar10.load_data()
    for i in range(10):
        _, labels = my_cifar10.train_loader[i]
        print(labels)
    # writer = SummaryWriter("tool")
    # for i, (x, y) in enumerate(my_cifar10.train_set):
    #     writer.add_images('input', x.unsqueeze(0), i)