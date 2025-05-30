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
|| Jittor     | Torch |
|:----:| :----: | :----: |
|版本| 1.3.9.14  | 1.12.0+cu116 |
|CPU| Paragraph   | Text        |

## 实验log
具体的日志见log文件夹
### 时间对比
|| Jittor     | Torch |
|:----:| :----: | :----: |
|数据加载| 1.3.9.14  | 1.12.0+cu116 |
|训练| Paragraph   | Text        |
### 性能对比
|| Jittor     | Torch |
|:----:| :----: | :----: |
|版本| 1.3.9.14  | 1.12.0+cu116 |
|CPU| Paragraph   | Text        |
---
trainloss曲线

---
acc-top1曲线
## DKD复现