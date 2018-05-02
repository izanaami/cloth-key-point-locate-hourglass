# 服饰关键点定位：Hourglass TensorFlow
Tensorflow implementation of Stacked Hourglass Networks for Human Pose Estimation by A.Newell et al.

Code as part of MSc Computing Individual Project (Imperial College London 2017)
## Based on
[Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937) -- A.Newell et al. 

Implentation of [Multi Context Attention Mechanism](https://arxiv.org/abs/1702.07432) -- Xiao Chu et al. -- Available (heavy model)
## 赛题与数据
见[FashionAI全球挑战赛——服饰关键点定位](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.4939d780MlsHSQ&raceId=231648)
## Hourglass model
模型源码参考自[wbenbihi/hourglasstensorlfow](https://github.com/wbenbihi/hourglasstensorlfow)
## version 2.0
5种类别分开训练，loss函数改为MSE
