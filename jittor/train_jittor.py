import os
import time
import torchvision.models as models
import jittor as jt
from jittor import nn, optim
from torch.utils.tensorboard import SummaryWriter
from config import TrainCfg
from tool.dataset_jittor import MyCIFAR10
from tool.utils_jittor import load_checkpoint, save_checkpoint, AverageMeter
from tool.utils_jittor import accuracy, validate, get_gt_mask, get_other_mask, cat_mask
from collections import OrderedDict
from tool.vgg_jittor import CIFAR10Quick, CIFAR10Simple
import jittor.nn as F


class BaseTrain:
    CE = 0
    KD = 1
    DKD = 2

    def __init__(self, config: TrainCfg, data: MyCIFAR10, modelset):
        self.cfg = config
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
        self.scheduler = jt.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg.MileStones,
            gamma=0.1
        )
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
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
            self.scheduler.last_epoch = epochs
        if self.loss_fn == self.KD or self.loss_fn == self.DKD:  # 蒸馏训练
            state_tmp = jt.load(self.cfg.TeacherDir)
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
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.model)
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
        lr = self.scheduler.get_lr()
        print("----lr----:", lr)
        print("----train loss----:", train_meters["losses"].avg)
        print("----acc top1----:", test_acc.item())
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
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
            self.optimizer.step(loss)
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
        soft_loss = self.kl_loss(log_pred_student, pred_teacher)
        soft_loss *= temperature ** 2
        hard_loss = nn.CrossEntropyLoss()(output, hard_label)
        print("soft loss:",soft_loss)
        return hard_loss + soft_loss

    def DKDloss_function(self, student, teacher, target, alpha, beta, temperature, warmup):
        gt_mask = get_gt_mask(student, target)
        other_mask = get_other_mask(student, target)
        # 得到全概率分布
        logit_student = F.softmax(student / temperature, dim=1)
        logit_teacher = F.softmax(teacher / temperature, dim=1)
        # 得到二分类概率分布
        binary_student = cat_mask(logit_student, gt_mask, other_mask)
        binary_teacher = cat_mask(logit_teacher, gt_mask, other_mask)
        log_binary_student = jt.log(binary_student)
        scale = temperature ** 2 # 尺度因子
        epsilon = 1e-10
        tckd_loss = self.kl_loss(log_binary_student, binary_teacher) * scale
        pred_teacher_part2 = F.softmax(teacher / temperature - 1000.0 * gt_mask, dim=1)+epsilon
        log_pred_student_part2 = F.log_softmax(student / temperature - 1000.0 * gt_mask, dim=1)
        nckd_loss = self.kl_loss(log_pred_student_part2, pred_teacher_part2) * scale
        soft_loss = alpha * tckd_loss + beta * nckd_loss
        hard_loss = nn.CrossEntropyLoss()(student, target)
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
        save_checkpoint(state, self.checkpoint_dir + "/epoch_{}.pkl".format(epoch))

    def save_bestpoint(self, state):
        save_checkpoint(state, self.checkpoint_dir + "/best.pkl")

    def load_checkpoint(self, filename):
        state = load_checkpoint(self.checkpoint_dir + filename)
        epochs = state["epoch"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.best_acc = state["best_acc"]
        return epochs


class TeacherTrain(BaseTrain):
    def __init__(self, config: TrainCfg, mydata: MyCIFAR10, model):
        super().__init__(config, mydata, model)
        self.loss_fn = self.CE
        self.checkpoint_dir = "../config/teacher"
        self.log_file = "../log/teacher/log.txt"
        self.writer = SummaryWriter("../log/teacher")


class KDTrain(BaseTrain):
    def __init__(self, config: TrainCfg, mydata: MyCIFAR10, model):
        super().__init__(config, mydata, model)
        self.loss_fn = self.KD
        self.checkpoint_dir = "../config/kd"
        self.log_file = "../log/kd/log.txt"
        self.writer = SummaryWriter("../log/kd")


class DKDTrain(BaseTrain):
    def __init__(self, config: TrainCfg, mydata: MyCIFAR10, model):
        super().__init__(config, mydata, model)
        self.loss_fn = self.DKD
        self.checkpoint_dir = "../config/dkd"
        self.log_file = "../log/dkd/log.txt"
        self.writer = SummaryWriter("../log/dkd")



if __name__ == '__main__':
    cfg = TrainCfg()
    my_cifar10 = MyCIFAR10(cfg.ratio)
    my_cifar10.stratified_sampling()
    my_cifar10.load_data()
    # 教师模型
    # model = CIFAR10Quick(num_classes=10)
    # trainer = TeacherTrain(cfg, my_cifar10, model)
    # trainer.train()
    # # 学生模型
    model1 = CIFAR10Simple()
    model1.load_state_dict(jt.load('../config/base.pkl')['model'])
    dkdtrainer = DKDTrain(cfg, my_cifar10, model1)
    dkdtrainer.train()

    # model2 = CIFAR10Simple()
    # model2.load_state_dict(jt.load('../config/base.pkl')['model'])
    # kdtrainer = KDTrain(cfg, my_cifar10, model2)
    # kdtrainer.train()
