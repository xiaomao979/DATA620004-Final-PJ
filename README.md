# Author
陈鸿旭 20110980002：报告  
李进之 20110980006：cutout和可视化  
员司雨 17307110448：cutmix和mixup算法  
# Introduction
DATA620004——神经网络和深度学习 期末project
对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-10或CIFAR-100图像分类任务中的性能表现。  

mixup: https://arxiv.org/pdf/1710.09412v2.pdf 

cutmix: https://arxiv.org/pdf/1905.04899.pdf

cutout: https://arxiv.org/pdf/1708.04552.pdf 

Note: 首先构建baseline方法(如AlexNet, ResNet-18)，在baseline的基础上分别加入上述不同的data augmentation方法。
注意：Baseline方法也需要对比

# Code
baseline
```
python main.py
```
mixup
```
python main_mixup.py
```
cutmix
```
python main_CutMix.py
```
cutout
```
python main_Cutout.py
```
#model
链接：https://pan.baidu.com/s/19Eta7c7J-oD4GLYEK4O3fQ 
提取码：glpq 
复制这段内容后打开百度网盘手机App，操作更方便哦
