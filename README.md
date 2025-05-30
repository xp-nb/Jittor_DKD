# DKD：Jittor与Torch对齐实验
## 环境配置
### 关于jittor安装
**在windows subsystem for linux中安装jittor。**
对于其他jittor安装方法，博主已经踩了不少坑：在windows中，jittor对编译器版本兼容性较差，编译失败。。。遂止；使用docker安装，不需要担心版本兼容性问题了，但是docker中的jittor版本还是4年前的1.2版本，其模块接口与官网文档不匹配，只能翻github的仓库对着源码写，而且这古早版本缺的算子太多了，要自己补很多，工程量太大。。。遂止；
使用wsl也并不是很省心。虽然wsl可以检测到gpu，但jittor没有检测到gpu,可能是cuda版本兼容性问题，博主不想再花精力安装了。。。好在cpu测试通过了，因此本项目的实验都是基于cpu进行的。
下面是jittor安装的版本与代码：
**Jittor(1.3.9.14)
g++(11.4.0)
cuda_driver_version: [12, 9]**
```shell
#wsl 下 jittor环境搭建
# 创建虚拟环境 jt python=3.7
# 安装编译器
sudo apt install g++ build-essential libomp-dev
# 激活虚拟环境
python3.7 -m venv jt
source jt/bin/activate
# 安装jittor
python3.7 -m pip install jittor
python3.7 -m jittor.test.test_example
```
### 实验环境
||   Torch   | Jittor |
|:----:| :----: | :----: |
|版本| 1.12.0+cu116 | 1.3.9.14  |
|CPU| i5-11400H| i5-13490F|

## 实验log
具体的日志见log文件夹
### 时间对比
|Teacher| Torch /s    |  Jittor /s|
|:----:| :----: | :----: |
|数据加载| 2.45 |  0.10 |
|训练|  5793.60   |    5696.30    |
|评估| 520.70|40.40|

|KD| Torch /s    |  Jittor /s|
|:----:| :----: | :----: |
|数据加载| 0.18 |  0.01 |
|训练|  463.38   |   220.13   |
|评估| 18.79|3.09|

|DKD| Torch /s    |  Jittor /s|
|:----:| :----: | :----: |
|数据加载| 0.19 |  0.01 |
|训练|  463.75   |    224.48    |
|评估| 18.66|3.05|

### 性能对比
|ACC-TOP1|    Torch  | Jittor |
|:----:| :----: | :----: |
|Teacher模型|  82.63 |  78.62( -4.01)|
|KD|   66.92  |  64.24( -2.68)    |
|DKD|66.59|64.30( -2.29)

在训练策略对齐下，jittor训练的teacher模型性能远逊色于（大约5%）torch，考虑到teacher准确率会影响蒸馏效果，因此teacher的训练策略并没有完全对齐,以此提高jittor的teacher性能

---
trainloss曲线

---
acc-top1曲线
## DKD复现
