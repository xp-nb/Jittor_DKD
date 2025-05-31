import os
import time
import torchvision.models as models
import torch
from torch import nn, optim
from torch.distributed._shard.checkpoint import load_state_dict
from torch.utils.tensorboard import SummaryWriter
from torch_.model.resnet_torch import resnet8x4, resnet32x4
from config import TrainCfg
from torch_.tool.dataset import MyCIFAR10
from torch_.tool.utils import adjust_learning_rate, load_checkpoint, save_checkpoint, AverageMeter
from torch_.tool.utils import accuracy, validate, _get_gt_mask, _get_other_mask, cat_mask
from collections import OrderedDict
from torch_.model.vgg_torch import VGG16, VGG11, load_pretrained_weights, CIFAR10Quick, CIFAR10Simple
import torch.nn.functional as F


class BaseTrain:
    CE = 0
    KD = 1
    DKD = 2

    def __init__(self, config: TrainCfg, data: MyCIFAR10, modelset):
        self.cfg = config
        self.device = torch.device("cpu")
        self.train_loader = data.train_loader
        self.val_loader = data.val_loader
        self.model = modelset
        self.teacher_model = CIFAR10Quick()
        self.loss_fn = self.CE
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LearningRate,
            momentum=0.9,
            weight_decay=0.0005,
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
        self.log_file = None
        self.checkpoint_dir = None
        self.writer = None

    def train(self, resume=False):
        epochs = 0
        if resume:  # 断点训练
            epochs = self.load_checkpoint("/best.pth")
        if self.loss_fn == self.KD or self.loss_fn == self.DKD:  # 蒸馏训练
            state_tmp = torch.load(self.cfg.TeacherDir)
            self.teacher_model.load_state_dict(state_tmp["model"])
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
        warmup = min(epoch / 10, 1.0)
        print("warmup:", warmup)
        self.train_iter(self.train_loader, train_meters, warmup)
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

    def train_iter(self, datas, train_meters, warmup):
        train_start_time = time.time()
        data_load_time = 0
        for data in datas:
            self.idx += 1
            data_start_time = time.time()
            image, target = data
            batch_size = image.shape[0]
            data_load_time += time.time() - data_start_time
            image = image.to(self.device)
            target = target.to(self.device)
            # forward
            out = self.model(image)
            if self.loss_fn == self.KD:
                teacher_out = self.teacher_model(image)
                # loss = self.DKDloss_function(out, teacher_out, target, 1.0, 1.0, self.cfg.Temperature, warmup)
                loss = self.KDloss_function(out, target, teacher_out, self.cfg.Temperature)
            elif self.loss_fn == self.DKD:
                teacher_out = self.teacher_model(image)
                loss = self.DKDloss_function(out, teacher_out, target, 1.0, 0.5, self.cfg.Temperature, warmup)
            else:
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

    def KDloss_function(self, output, hard_label, teacher, temperature):
        log_pred_student = F.log_softmax(output / temperature, dim=1)
        pred_teacher = F.softmax(teacher / temperature, dim=1)
        soft_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        soft_loss *= temperature ** 2
        hard_loss = nn.CrossEntropyLoss()(output, hard_label)
        print("hard_loss:", hard_loss)
        print("soft_loss:", soft_loss)
        return hard_loss + soft_loss

    def DKDloss_function(self, logits_student, logits_teacher, target, alpha, beta, temperature, warmup):
        # 解耦
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        soft_loss = alpha * tckd_loss + beta * nckd_loss
        hard_loss = nn.CrossEntropyLoss()(logits_student, target)

        return hard_loss + warmup*soft_loss

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


def DKDloss_function( logits_student, logits_teacher, target, alpha, beta, temperature, warmup):
    # 解耦
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    print(pred_teacher_part2)
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    print(log_pred_student_part2)
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    print(nckd_loss)
    soft_loss = alpha * tckd_loss + beta * nckd_loss
    hard_loss = nn.CrossEntropyLoss()(logits_student, target)
    print(hard_loss)
    return hard_loss + warmup*soft_loss

class TeacherTrain(BaseTrain):
    def __init__(self, config: TrainCfg, mydata: MyCIFAR10, model):
        super().__init__(config, mydata, model)
        self.loss_fn = self.CE
        self.checkpoint_dir = "../config/teacher"
        self.log_file = "../log/teacher/teacher_log.txt"
        self.writer = SummaryWriter("../log/teacher")


class KDTrain(BaseTrain):
    def __init__(self, config: TrainCfg, mydata: MyCIFAR10, model):
        super().__init__(config, mydata, model)
        self.loss_fn = self.KD
        self.checkpoint_dir = "../config/student_kd"
        self.log_file = "../log/student/KD/log.txt"
        self.writer = SummaryWriter("../log/student/KD")


class DKDTrain(BaseTrain):
    def __init__(self, config: TrainCfg, mydata: MyCIFAR10, model):
        super().__init__(config, mydata, model)
        self.loss_fn = self.DKD
        self.checkpoint_dir = "../config/student_dkd"
        self.log_file = "../log/student/DKD/log.txt"
        self.writer = SummaryWriter("../log/student/DKD")


if __name__ == '__main__':
    cfg = TrainCfg()
    my_cifar10 = MyCIFAR10(cfg)
    my_cifar10.stratified_sampling()
    my_cifar10.load_data()
    # # 教师模型
    # # model = CIFAR10Quick(num_classes=10)
    # # trainer = TeacherTrain(cfg, my_cifar10, model)
    # # 学生模型
    # model1 = CIFAR10Simple()
    # model1.load_state_dict(torch.load('../config/base.pth')['model'])
    # dkdtrainer = DKDTrain(cfg, my_cifar10, model1)
    # dkdtrainer.train()

    model2 = CIFAR10Simple()
    model2.load_state_dict(torch.load('../config/base.pth')['model'])
    dkdtrainer = KDTrain(cfg, my_cifar10, model2)
    dkdtrainer.train()

    # sl = torch.asarray([[3,4,5],[3,4,5]], dtype=torch.float32)
    # tl = torch.asarray([[3,9,9],[3,9,9]], dtype=torch.float32)
    # target = torch.asarray([2,2], dtype=torch.long)
    # result = DKDloss_function(sl,tl,target,1,2,4,1)
