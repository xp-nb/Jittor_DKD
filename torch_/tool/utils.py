import os
import torch
import torch.nn as nn
import numpy as np
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, model, device):
    losses, top1, top5 = [AverageMeter() for _ in range(3)]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for image, target in val_loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss, batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
    return top1.avg, top5.avg, losses.avg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.LrDecayStages))
    if steps > 0:
        decay_rate = 0.1 ** steps  # 使用固定衰减率0.1的steps次方
        new_lr = float(cfg.LearningRate * decay_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return float(cfg.LearningRate)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    torch.save(obj, path)


def load_checkpoint(path):
    print("Loading checkpoint from {}".format(path))
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


