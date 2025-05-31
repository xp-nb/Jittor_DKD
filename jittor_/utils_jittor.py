import jittor as jt
import jittor.nn as nn
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


def validate(val_loader, model):
    losses, top1, top5 = [AverageMeter() for _ in range(3)]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with jt.no_grad():
        start_time = time.time()
        for image, target in val_loader:
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss, batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
    return top1.avg, top5.avg, losses.avg



def accuracy(output, target, topk=(1,)):
    with jt.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred == target.reshape(1, -1).expand_as(pred)  # 修改此处
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    jt.save(obj, path)


def load_checkpoint(path):
    print("Loading checkpoint from {}".format(path))
    return jt.load(path)


def get_gt_mask(logits, target):
    """得到目标掩码"""
    target = target.reshape(-1).unsqueeze(1)
    ones = jt.ones_like(target)
    zeros = jt.zeros_like(logits)
    mask = jt.scatter(zeros,1, target, ones).bool()
    return mask


def get_other_mask(logits, target):
    """得到非目标掩码"""
    target = target.reshape(-1).unsqueeze(1)
    ones = jt.ones_like(logits)
    zeros = jt.zeros_like(target)
    mask = jt.scatter(ones,1, target, zeros).bool()
    return mask

def cat_mask(t, mask1, mask2):
    """组合二分类概率分布"""
    t1 = (t * mask1).sum(dim=1, keepdims=True).squeeze()
    t2 = (t * mask2).sum(1, keepdims=True).squeeze()
    result = jt.stack((t1, t2),dim=1)
    return result

if __name__ == '__main__':
    logits = jt.array([[0.1,0.2,0.4,0.3],[0.1,0.2,0.4,0.3],[0.1,0.2,0.4,0.3]])
    logits1 = jt.array([[0.1,0.3,0.3,0.3],[0.1,0.2,0.4,0.3],[0.1,0.2,0.4,0.3]])
    targets = jt.array([1,2,3],dtype=jt.float32)
    mask1 = get_gt_mask(logits, targets)
    print(logits-1000*mask1)
    mask2 = get_other_mask(logits, targets)
    # target = targets.reshape(-1).unsqueeze(1)
    # print(target.data)
    # mask = jt.zeros_like(logits)
    # print(mask.data)
    # mask = jt.scatter(mask, 1, target, jt.ones_like(target)).bool()
    re = cat_mask(logits,mask1,mask2)
    klv = jt.nn.KLDivLoss(reduction='mean')
    loss = klv(jt.log(logits),logits1)
    print(loss)