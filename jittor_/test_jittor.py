import jittor as jt

from jittor_.vgg_jittor import CIFAR10Simple,CIFAR10Quick
from jittor_.config import TrainCfg
from jittor_.dataset_jittor import MyCIFAR10

if __name__ == '__main__':
    teacher_path = "../config/teacher.pkl" #  best:78.18
    kd_path = "../config/kd.pkl"
    dkd_path = "../config/dkd.pkl"

    model_path = dkd_path

    if model_path == teacher_path:
        model = CIFAR10Quick()
    else:
        model = CIFAR10Simple()
    model.load_state_dict(jt.load(model_path)['model'])
    print("模型加载成功")
    cfg = TrainCfg()
    cfg.ratio = 1
    my_cifar10 = MyCIFAR10(cfg.ratio,train=False)
    my_cifar10.stratified_sampling()
    my_cifar10.load_data()
    batch_num = 0
    total_acc = 0
    for i, (images, labels) in enumerate(my_cifar10.val_loader):
        output = model(images)
        predicted = jt.argmax(output.data, 1)
        total_num = len(labels)
        correct = (predicted[0].data == labels).sum().item()
        accuracy = correct / total_num
        total_acc += accuracy
        batch_num += 1
        print("第{}批次，model准确率为：{:.4f}".format(i + 1, accuracy))
    print("Total model准确率为：{:.4f}".format(total_acc/batch_num))