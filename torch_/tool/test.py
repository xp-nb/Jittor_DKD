import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch_.model.vgg_torch import CIFAR10Simple,CIFAR10Quick
from config import TrainCfg
from dataset import MyCIFAR10

if __name__ == '__main__':
    teacher_path = "../config/teacher/teacher.pth"
    kd_path = "../config/student_kd/best.pth"
    dkd_path = "../config/student_dkd/best.pth"
    batch_size = 128
    kd_model = CIFAR10Simple()
    kd_model.load_state_dict(torch.load(kd_path)['model'])
    dkd_model = CIFAR10Simple()
    dkd_model.load_state_dict(torch.load(dkd_path)['model'])
    teacher_model = CIFAR10Quick()
    teacher_model.load_state_dict(torch.load(teacher_path)['model'])
    print("模型加载成功")
    cfg = TrainCfg()
    cfg.ratio = 1
    my_cifar10 = MyCIFAR10(cfg,train=False)
    my_cifar10.stratified_sampling()
    my_cifar10.load_data()
    batch_num = 0
    total_acc_kd = 0
    total_acc_dkd = 0
    total_acc_teacher = 0
    for i, (images, labels) in enumerate(my_cifar10.val_loader):
        images = images.to('cpu')
        output_kd = kd_model(images)
        output_dkd = dkd_model(images)
        output_teacher = teacher_model(images)
        _, predicted_kd = torch.max(output_kd.data, 1)
        _, predicted_dkd = torch.max(output_dkd.data, 1)
        _, predicted_teacher = torch.max(output_teacher.data, 1)
        total_num = len(labels)
        print(total_num)
        correct_kd = (predicted_kd == labels).sum().item()
        correct_dkd = (predicted_dkd == labels).sum().item()
        correct_teacher = (predicted_teacher == labels).sum().item()
        accuracy_kd = correct_kd / total_num
        accuracy_dkd = correct_dkd / total_num
        accuracy_teacher = correct_teacher / total_num
        total_acc_kd += accuracy_kd
        total_acc_dkd += accuracy_dkd
        total_acc_teacher += accuracy_teacher
        batch_num += 1
        print("第{}批次，kd准确率为：{:.4f},dkd准确率为：{:.4f}，teacher准确率为：{:.4f}"
              .format(i + 1, accuracy_kd,accuracy_dkd, accuracy_teacher))
    print("Total kd准确率为：{:.4f},dkd准确率为：{:.4f}，teacher准确率为：{:.4f}"
          .format(total_acc_kd/batch_num, total_acc_dkd/batch_num, total_acc_teacher/batch_num))