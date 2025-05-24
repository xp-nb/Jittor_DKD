import os
import time
import torchvision.models as models
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from model.resnet_torch import resnet8x4, resnet32x4
from config import TrainCfg
from tool.dataset import MyCIFAR10
from tool.utils import adjust_learning_rate, load_checkpoint, save_checkpoint, AverageMeter, accuracy, validate
from collections import OrderedDict
from model.vgg_torch import VGG16, VGG11, load_pretrained_weights,CIFAR10Quick


class BaseTrain:
    def __init__(self, config: TrainCfg, data: MyCIFAR10, modelset):
        self.cfg = config
        self.device = torch.device("cuda")
        self.train_loader = data.train_loader
        self.val_loader = data.val_loader
        self.model = modelset
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LearningRate,
            # momentum = 0.9,
            # weight_decay = 0.0005,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg.MileStones,
            gamma=0.1
        )
        self.idx = 0
        self.best_acc = -1
        # log
        self.total_dataload = 0
        self.total_traintime = 0
        self.total_evaltime = 0
        self.writer = SummaryWriter("../log/teacher")
        self.checkpoint_dir = "../config/teacher"
        self.log_file = "../log/teacher/teacher_log.txt"

    def train(self, resume=False):
        epochs = 0
        if resume:  # 断点训练
            epochs = self.load_checkpoint("/best.pth")

        while epochs < self.cfg.Epoch:
            epochs += 1
            self.train_epoch(epochs)
        with open(self.log_file, "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)) + os.linesep)
            writer.write("total_dataload\t" + "{:.2f}".format(float(self.total_dataload)) + "secends" + os.linesep)
            writer.write("total_traintime\t" + "{:.2f}".format(float(self.total_traintime)) + "secends" + os.linesep)
            writer.write("total_evaltime\t" + "{:.2f}".format(float(self.total_evaltime)) + "secends" + os.linesep)

    def train_epoch(self, epoch):
        # lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        self.model.train()
        print("epoch:", epoch)
        self.train_iter(self.train_loader, train_meters)
        start_time = time.time()
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.model, self.device)
        self.total_evaltime += time.time() - start_time
        self.writer.add_scalar("train_top1", train_meters["top1"].avg, epoch)
        self.writer.add_scalar("test_top1", test_acc.item(), epoch)
        self.writer.add_scalar("traing_loss", train_meters["losses"].avg, epoch)
        self.writer.add_scalar("test_loss", test_loss.item(), epoch)
        # 记录日志并保存检查点
        log_dict = OrderedDict(
            {
                "train_acc_top1": train_meters["top1"].avg,
                "train_acc_top5": train_meters["top5"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc_top1": test_acc.item(),
                "test_acc_top5": test_acc_top5.item(),
                "test_loss": test_loss.item(),
            }
        )
        lr = self.scheduler.get_last_lr()[0]
        print("----lr----:", lr)
        print("----train loss----:", train_meters["losses"].avg)
        print("----acc top1----:", test_acc.item())
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_acc": self.best_acc,
        }
        if epoch % self.cfg.SaveCheckpoint == 0:
            self.save_epochpoint(epoch, state)
        if test_acc >= self.best_acc:
            self.best_acc = test_acc
            self.save_bestpoint(state)
        self.log(lr, epoch, log_dict)

    def train_iter(self, datas, train_meters):
        batch_size = self.cfg.BatchSize
        train_start_time = time.time()
        data_load_time = 0
        for data in datas:
            self.idx += 1
            data_start_time = time.time()
            image, target = data
            data_load_time += time.time() - data_start_time
            image = image.to(self.device)
            target = target.to(self.device)
            # forward
            out = self.model(image)
            loss = self.loss_function(out, target)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            top1, top5 = accuracy(out, target, (1, 5))
            train_meters["losses"].update(loss.item(), batch_size)
            train_meters["top1"].update(top1.item(), batch_size)
            train_meters["top5"].update(top5.item(), batch_size)
        self.scheduler.step()
        self.total_traintime += time.time() - train_start_time
        self.total_dataload += data_load_time
        print("time：", time.time() - train_start_time)

    def loss_function(self, output, target):
        return nn.CrossEntropyLoss()(output, target)

    def log(self, lr, epoch, log_dict):
        # worklog.txt
        with open(self.log_file, "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.4f}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    def save_epochpoint(self, epoch, state):
        save_checkpoint(state, self.checkpoint_dir + "/epoch_{}.pth".format(epoch))

    def save_bestpoint(self, state):
        save_checkpoint(state, self.checkpoint_dir + "/best.pth")

    def load_checkpoint(self, filename):
        state = load_checkpoint(self.checkpoint_dir + filename)
        epochs = state["epoch"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.best_acc = state["best_acc"]
        return epochs


class TeacherTrain(BaseTrain):
    def __init__(self, config: TrainCfg, mydata: MyCIFAR10, model):
        super().__init__(config, mydata, model)
        self.cfg = config
        self.device = torch.device("cpu")
        self.model = model
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LearningRate,
            # momentum = 0.9,
            # weight_decay = 0.0005,
        )
        self.loss_func = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.best_acc = -1
        self.train_loader = mydata.train_loader
        self.val_loader = mydata.val_loader
        self.writer = SummaryWriter("../log/teacher")


if __name__ == '__main__':
    cfg = TrainCfg()
    my_cifar10 = MyCIFAR10(cfg)
    my_cifar10.stratified_sampling()
    my_cifar10.load_data()
    model = CIFAR10Quick(num_classes=10)
    # model.load_state_dict(load_pretrained_weights("../config/teacher/vgg11.pth"), strict=False)
    trainer = TeacherTrain(cfg, my_cifar10, model)
    trainer.train()
