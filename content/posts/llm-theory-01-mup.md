---
date: '2026-05-04T14:10:04+08:00'
draft: false
title: 'LLM Theory 01: MuP'
categories: ["AI"]
tags: ["LLM", "Pretraining", "MuP", "Hyperparameter Transfer"]
series: ["LLM Theory"]
series_order: 1
weight: 1
math: true
---

> 这篇文章是关于 LLM 预训练中 MuP / μP 的学习笔记，主要参考原论文
> [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
> 和苏剑林的博客
> [《初探MuP：超参数的跨模型尺度迁移规律》](https://spaces.ac.cn/archives/10770)。

## 问题背景

LLM 预训练的成本很高，因此我们通常不希望直接在目标规模的大模型上反复搜索学习率、初始化、权重衰减等超参数。一个自然想法是：先在同架构的小模型上调参，再把超参数迁移到大模型上。

MuP（Maximal Update Parametrization）的核心目标，就是让某些关键超参数在模型宽度变化时尽量保持稳定，从而支持从小模型到大模型的 zero-shot hyperparameter transfer。原论文把这个迁移范式称为 μTransfer。

因为苏老师的博客写得太简洁+深入浅出了，本文也不会重复去讲他讲的很完善的部分，而是对他的内容进行补充和完善。个人在读原博客的时候感觉到一些地方苏老师略过了一些思考过程，导致初次理解时会让人觉得有些跳跃。

## 正文

MuP问题的出发点很简单，模型是一个黑盒，因此想训练出一个好模型，无法避免地要做大量的尝试（也就是俗称的调参炼丹）。但对于大模型而言，尝试的时间&金钱&人力成本很高。MuP就是对传统的炼丹过程做了一个剪枝，通过数学推导证明了小模型上已经被验证的某些规律可以直接扩展到大模型上。

当然，上述的总结是比较泛化的，具体到实践中，肯定还会问几个问题：模型大小如何界定？哪些规律可以扩展？具体如何扩展？在展开具体方法之前笔者可以回答前两个问题，这里模型的大小用神经网络的宽度/隐藏层维度来量化。可扩展的规律主要指学习率的选择。因此MuP解决的具体问题是“在网络加宽的情况下，学习率应该如何跟随着隐藏层维度改变”。

## 模型宽度的影响：无法尽善尽美的参数初始化
为什么要先讲参数初始化呢，因为参数初始化提供了一个最基础的视角，来定量描述“宽度影响稳定性”这件事。

前传和反传的最优参数初始化无法兼容。
高维的任意两个向量夹角都是几乎正交的，可以算一下任意向量和单位向量的夹角，这里不赘述。
所以苏剑林老师基于这点给了一个推论：
从$N(0,1/n)$
中随机选取$n^2$
个数，组成一个$n×n$
的矩阵，这个矩阵近似为正交矩阵，且$n$
越大，近似程度越好。
其实道理是一样的，列向量两两正交，就是正交矩阵。n越大，相当于维度越高，正交概率越大。每个向量里面的元素都是采样出来的，所以每个元素的值大约是`sqrt(1/n)`，所以整个向量的模长平方就是$n * (sqrt(1/n))^2 = 1$

正交矩阵有一个好的性质，就是它作用于一个向量时，不改变向量模长。神经网络是对一个输入向量做很多次变换，
得到一个输出向量。我们希望输入向量在变换为输出向量的游走过程中，能一直在一个球面上，也就是模长不变。因为这样
从直觉上可以大幅压缩向量遍历的空间。可以想象一下，在一个完整的高维空间里面找最优解,和在空间内的一个球面
找最优解显然是后者更容易。如果向量变换前后都在同一个球面上或者近似在一个厚度比较薄的球壳上，本文将这种性质称为“稳定性”。

所以最经典的初始化方式是推论里面的采样方式。上述结论也可以通过让变换前后的RMS相等来推导。如果引入了激活函数，初始化的值略有不同，但是推导逻辑类似。

前传和反传区别不大，都是矩阵乘。

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{X}}
\sim
\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}
\boldsymbol{W}^{\top}
$$

主要的尺度变化也来自于$W$。输入和输出的维度不相等的时候，就找不到一个两全其美的采样方差。这是一个open的问题，苏老师在原文这里提出这个问题并不是为了直接解决这个问题，而是为了说明模型的宽度和中间层稳定性之间存在着直接的关系。

## Loss的稳定性

在苏老师这篇MuP的博客中透露着一个隐含的insight：模型加宽带来的难度就是稳定性下降。（也许这个insight来自于训模型时候的经验）这个稳定性可以是Loss的稳定性，也可以是梯度的稳定性，还可以是每一层输出结果的稳定性。这里考虑了损失增量的稳定性。

文章中需要推导或注释的地方有两点，第一点是公式6如何近似，第二点是公式4如何得到公式7。

公式6的近似：
$$
\Delta \mathcal{L}=\mathcal{L}(\boldsymbol{W}+\Delta \boldsymbol{W})-\mathcal{L}(\boldsymbol{W})
$$
一阶泰勒近似：
$$
\mathcal{L}(\boldsymbol{W}+\Delta \boldsymbol{W})\approx\mathcal{L}(\boldsymbol{W})+\sum_{i,j}\frac{\partial \mathcal{L}}{\partial W_{ij}}\Delta W_{ij}
$$

$$
\Delta \mathcal{L}\approx\sum_{i,j}\frac{\partial \mathcal{L}}{\partial W_{ij}}\Delta W_{ij}
$$

$$
\langle \boldsymbol{A},\boldsymbol{B}\rangle_F=\sum_{i,j} A_{ij}B_{ij}
$$

$$
\sum_{i,j}\frac{\partial \mathcal{L}}{\partial W_{ij}}\Delta W_{ij}=\left\langle\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}},\Delta \boldsymbol{W}\right\rangle_F
$$

$$
\Delta \mathcal{L}=\mathcal{L}(\boldsymbol{W}+\Delta \boldsymbol{W})-\mathcal{L}(\boldsymbol{W})\approx\left\langle\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}},\Delta \boldsymbol{W}\right\rangle_F
$$

7的推导
公式4的近似：

$$
\boldsymbol{Y}=\phi(\boldsymbol{X}\boldsymbol{W})
$$

令：

$$
\boldsymbol{Z}=\boldsymbol{X}\boldsymbol{W}
$$

则：

$$
\boldsymbol{Y}=\phi(\boldsymbol{Z})
$$

根据链式法则：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}=
\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}\otimes\phi'(\boldsymbol{Z})
$$

又因为：

$$
\boldsymbol{Z}=\boldsymbol{X}\boldsymbol{W}
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}=\boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}=\boldsymbol{X}^{\top}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}\otimes\phi'(\boldsymbol{X}\boldsymbol{W})\right)
$$

由于常见激活函数的导数 \(\phi'\) 通常是常数尺度，因此做数量级分析时可以近似为：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}
\sim
\boldsymbol{X}^{\top}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}
$$

公式4的近似：

$$
\boldsymbol{Y}=\phi(\boldsymbol{X}\boldsymbol{W})
$$

令：

$$
\boldsymbol{Z}=\boldsymbol{X}\boldsymbol{W}
$$

则：

$$
\boldsymbol{Y}=\phi(\boldsymbol{Z})
$$

根据链式法则：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}\otimes\phi'(\boldsymbol{Z})
$$

又因为：

$$
\boldsymbol{Z}=\boldsymbol{X}\boldsymbol{W}
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}=\boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}
$$

代入 \(\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}\)：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}=\boldsymbol{X}^{\top}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}\otimes\phi'(\boldsymbol{X}\boldsymbol{W})\right)
$$

由于常见激活函数的导数 \(\phi'\) 通常是常数尺度，因此做数量级分析时可以近似为：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}\sim\boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}
$$

公式7的推导：

$$
\Delta \mathcal{L}\approx\left\langle\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}},\Delta \boldsymbol{W}\right\rangle_F
$$

梯度下降更新为：

$$
\Delta \boldsymbol{W}=-\eta\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}
$$

代入上式：

$$
\Delta \mathcal{L}\approx\left\langle\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}},-\eta\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}\right\rangle_F
$$

$$
\Delta \mathcal{L}\approx-\eta\left\langle\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}},\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}\right\rangle_F
$$

根据 Frobenius 范数定义：

$$
\left\langle\boldsymbol{A},\boldsymbol{A}\right\rangle_F=\Vert \boldsymbol{A}\Vert_F^2
$$

所以：

$$
\Delta \mathcal{L}\approx-\eta\left\Vert\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}\right\Vert_F^2
$$

再由公式4：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}\sim\boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}
$$

因此：

$$
\left\Vert\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}\right\Vert_F^2\sim\left\Vert\boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}\right\Vert_F^2
$$

最终得到：

$$
\Delta \mathcal{L}\approx-\eta\left\Vert\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}\right\Vert_F^2\sim-\eta\left\Vert\boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}\right\Vert_F^2
$$

最后这个公式就很明显了，如果η保持不变，不要说把模型变宽，甚至同一个模型的不同层都会导致稳定性下降。因为一个模型的不同层维度也不一定一样。
如果让每一层的学习率都自适应于dim_in × dim_out理论上可以，但是又会出现之前前向反向不兼容的问题。
为此其实就有一个trick，虽然原文中没明确强调，但是隐含在后面的推导之中：
中间层的in和out维度相同，保证不会出现前传反传的兼容问题，这样问题就可以暂时简化了，只需要考虑最外部的indim和最外部的outdim（这两个是由外部决定的，不能随便改）和中间层维度dim。
中间层根据dim自适应调整的策略我们已经推导了，剩下的问题就只是处理最开始和最后一层。而对他们特殊处理的方式也不难，只需要想办法在RMS=theta 1的时候凑出来学习率关于dim的表达式即可。

## 不同层的RMS

其实上一节已经把主要问题解决干净了，最后剩下的就是一些实际处理上的细节。原文把神经网络权重划分成三类：in，out，hidden。in和out之所以单独拿出来，是因为它们有一个维度是尺度无关的，所以RMS算出来和中间层不一样。
因此本文只是把RMS计算的不同展开写一下。

RMS 和 Frobenius 范数的关系：

$$
\mathrm{RMS}(\boldsymbol{A})=\sqrt{\frac{1}{mn}\sum_{i,j}A_{ij}^2}
$$

$$
\|\boldsymbol{A}\|_F^2=\sum_{i,j}A_{ij}^2
$$

$$
\|\boldsymbol{A}\|_F^2=mn\cdot \mathrm{RMS}(\boldsymbol{A})^2
$$

输出层梯度：

$$
\boldsymbol{Z}=\boldsymbol{Y}_{out}\boldsymbol{W}_{out}
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{out}}=\boldsymbol{Y}_{out}^{\top}\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}
$$

$$
\mathrm{RMS}(\boldsymbol{Y}_{out})=\Theta(1)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}\right)=\Theta(1)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{out}}\right)=\Theta(1)
$$

$$
\boldsymbol{W}_{out}\in\mathbb{R}^{d\times d_{out}}
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{out}}\right\|_F^2=d\cdot d_{out}\cdot\Theta(1)^2
$$

$$
d_{out}=\Theta(1)
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{out}}\right\|_F^2=\Theta(d)
$$

$$
\Delta \mathcal{L}_{out}\approx-\eta_{out}\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{out}}\right\|_F^2
$$

$$
\Delta \mathcal{L}_{out}=\Theta(1)\Rightarrow \eta_{out}\Theta(d)=\Theta(1)
$$

$$
\eta_{out}\propto\frac{1}{d}
$$

中间层梯度：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{out}}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}\boldsymbol{W}_{out}^{\top}
$$

$$
\mathrm{Var}(\boldsymbol{W}_{out})\propto\frac{1}{d^2}
$$

$$
\mathrm{RMS}(\boldsymbol{W}_{out})=\Theta\left(\frac{1}{d}\right)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}\right)=\Theta(1)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{out}}\right)=\Theta\left(\frac{1}{d}\right)
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_k}=\frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{W}_k}\cdot\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{out}}
$$

$$
\mathrm{RMS}\left(\frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{W}_k}\right)=\Theta(1)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_k}\right)=\Theta(1)\cdot\Theta\left(\frac{1}{d}\right)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_k}\right)=\Theta\left(\frac{1}{d}\right)
$$

$$
\boldsymbol{W}_k\in\mathbb{R}^{d\times d}
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_k}\right\|_F^2=d^2\cdot\Theta\left(\frac{1}{d}\right)^2
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_k}\right\|_F^2=d^2\cdot\Theta\left(\frac{1}{d^2}\right)
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_k}\right\|_F^2=\Theta(1)
$$

$$
\Delta \mathcal{L}_k\approx-\eta_k\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_k}\right\|_F^2
$$

$$
\Delta \mathcal{L}_k=\Theta(1)\Rightarrow \eta_k\Theta(1)=\Theta(1)
$$

$$
\eta_k=\Theta(1)
$$

输入层梯度：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{in}}=\boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{in}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{in}}=\frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{Y}_{in}}\cdot\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{out}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{out}}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{Z}}\boldsymbol{W}_{out}^{\top}
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{out}}\right)=\Theta\left(\frac{1}{d}\right)
$$

$$
\mathrm{RMS}\left(\frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{Y}_{in}}\right)=\Theta(1)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{in}}\right)=\Theta(1)\cdot\Theta\left(\frac{1}{d}\right)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}_{in}}\right)=\Theta\left(\frac{1}{d}\right)
$$

$$
\mathrm{RMS}(\boldsymbol{X})=\Theta(1)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{in}}\right)=\Theta\left(\frac{1}{d}\right)
$$

$$
\boldsymbol{W}_{in}\in\mathbb{R}^{d_{in}\times d}
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{in}}\right\|_F^2=d_{in}\cdot d\cdot\Theta\left(\frac{1}{d}\right)^2
$$

$$
d_{in}=\Theta(1)
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{in}}\right\|_F^2=\Theta(d)\cdot\Theta\left(\frac{1}{d^2}\right)
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{in}}\right\|_F^2=\Theta\left(\frac{1}{d}\right)
$$

$$
\Delta \mathcal{L}_{in}\approx-\eta_{in}\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{in}}\right\|_F^2
$$

$$
\Delta \mathcal{L}_{in}=\Theta(1)\Rightarrow \eta_{in}\Theta\left(\frac{1}{d}\right)=\Theta(1)
$$

$$
\eta_{in}\propto d
$$

总结：

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{out}}\right)=\Theta(1)
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{out}}\right\|_F^2=\Theta(d)
$$

$$
\eta_{out}\propto\frac{1}{d}
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_k}\right)=\Theta\left(\frac{1}{d}\right)
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_k}\right\|_F^2=\Theta(1)
$$

$$
\eta_k=\Theta(1)
$$

$$
\mathrm{RMS}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{in}}\right)=\Theta\left(\frac{1}{d}\right)
$$

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_{in}}\right\|_F^2=\Theta\left(\frac{1}{d}\right)
$$

$$
\eta_{in}\propto d
$$

这里需要注意的是，结果之所以如此简洁，是因为假设神经网络都是方阵，因此不存在前述的in out维度冲突问题。现实中落地MuP的时候，并不是尽量让网络设置成方阵，而是需要先定义好：模型的哪些层的宽度是随着参数量增大而线性增大的。对这些层应用MuP即可，而这些层（矩阵）又被分为三类，分别是被in/out限制住一维，或两维度都随着宽度线性增长。

那这里还可以提一个问题，原文里面的权重随尺度增长的形式都是$d×4d$ or $d × const$这种形式，那如果是$d×d^2$或者$d^p×d^q$这种形式呢？（太变态了bro，谁会这么设计模型-.-）这样搞带来最大的问题就是前向和反向的尺度不一样了，对这一层而言in的尺度是p次方而out的尺度是q次方，前向反向永远是不兼容的，因为d不同阶了。写成公式就是：


$$
\boldsymbol{W}\in\mathbb{R}^{\Theta(d^p)\times\Theta(d^q)}
$$

如果：

$$
\mathrm{RMS}\left(\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}}\right)=\Theta(d^{-s})
$$

那么：

$$
\left\|\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}}\right\|_F^2=\Theta(d^{p+q-2s})
$$

为了让：

$$
\Delta\mathcal{L}=\Theta(1)
$$

需要：

$$
\eta=\Theta\left(d^{-(p+q-2s)}\right)
$$

所以：

$$
a(d)=d^2
$$

当然会改变大 \(\Theta\) 计算。

## 保持稳定性只需要约束loss变化量吗
这又是苏老师原文中第二个比较跳跃的点，虽然推导出MuP策略就是单纯使用约束Loss变化量搞出来的，但是实际上还是不够严谨，正确的推导应该是Loss变化量约束+一个什么其他的条件，才能得到MuP。从数学上苏老师原文已经说了为什么不严谨，笔者这里从直觉上提供一个简单的说明：只约束loss变化量显然是不够的，因为loss是一个标量，向量转化成标量的时候丢失了一些信息，这个就让我们不确定向量是如何更新的，只能看到loss的数值保持尺度不变，但是权重不一定尺度不变。

因此还加上了一个约束，就是特征图的变化也要有尺度不变。尺度不变当然还是要用RMS来描述~

下面是补充的证明过程：

$$
\Delta\mathcal{L}=\Theta(1)
$$

每一层输出，也就是 feature / activation 的变化量，也要保持尺度稳定。

$$
\mathrm{RMS}(\Delta\boldsymbol{Y}_k)=\Theta(1)
$$

对于一个线性层：

$$
\boldsymbol{Y}_k=\boldsymbol{Y}_{k-1}\boldsymbol{W}_k
$$

如果参数更新为：

$$
\boldsymbol{W}_k\rightarrow\boldsymbol{W}_k+\Delta\boldsymbol{W}_k
$$

那么输出变化为：

$$
\Delta\boldsymbol{Y}_k=\boldsymbol{Y}_{k-1}(\boldsymbol{W}_k+\Delta\boldsymbol{W}_k)-\boldsymbol{Y}_{k-1}\boldsymbol{W}_k
$$

所以：

$$
\Delta\boldsymbol{Y}_k=\boldsymbol{Y}_{k-1}\Delta\boldsymbol{W}_k
$$

其中：

$$
\boldsymbol{Y}_{k-1}\in\mathbb{R}^{b\times d}
$$

$$
\Delta\boldsymbol{W}_k\in\mathbb{R}^{d\times d}
$$

于是：

$$
\Delta\boldsymbol{Y}_k\in\mathbb{R}^{b\times d}
$$

展开其中一个元素：

$$
(\Delta\boldsymbol{Y}_k)_{ij}=\sum_{\ell=1}^{d}(\boldsymbol{Y}_{k-1})_{i\ell}(\Delta\boldsymbol{W}_k)_{\ell j}
$$

如果 \(\Delta\boldsymbol{W}_k\) 和 \(\boldsymbol{Y}_{k-1}\) 是独立随机的，那么这个内积会有随机正负抵消，量级可能是：

$$
\mathrm{RMS}(\Delta\boldsymbol{Y}_k)\sim\sqrt{d}\cdot\mathrm{RMS}(\Delta\boldsymbol{W}_k)
$$

但这里的 \(\Delta\boldsymbol{W}_k\) 不是初始化的随机矩阵，而是由梯度下降得到的更新量：

$$
\Delta\boldsymbol{W}_k=-\eta_k\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}_k}
$$

所以 \(\Delta\boldsymbol{W}_k\) 和 \(\boldsymbol{Y}_{k-1}\) 不应当被看作完全独立。作者认为它们之间有较强相关性，因此这个 \(d\) 维内积更接近 \(d\) 项同向累加的量级：

$$
\mathrm{RMS}(\Delta\boldsymbol{Y}_k)\sim d\cdot\mathrm{RMS}(\Delta\boldsymbol{W}_k)
$$

为了让特征变化保持稳定：

$$
\mathrm{RMS}(\Delta\boldsymbol{Y}_k)=\Theta(1)
$$

就需要：

$$
d\cdot\mathrm{RMS}(\Delta\boldsymbol{W}_k)=\Theta(1)
$$

因此：

$$
\mathrm{RMS}(\Delta\boldsymbol{W}_k)=\Theta\left(\frac{1}{d}\right)
$$

这就是特征变化条件给出的关键约束。


假设输出层权重的 RMS 是：

$$
\mathrm{RMS}(\boldsymbol{W}_{out})=\Theta(d^{-\alpha})
$$

那么对应的方差是：

$$
\mathrm{Var}(\boldsymbol{W}_{out})=\Theta(d^{-2\alpha})
$$

因为：

$$
\mathrm{RMS}(\boldsymbol{W}_{out})\approx\sqrt{\mathrm{Var}(\boldsymbol{W}_{out})}
$$

输出层反传到 hidden feature 的梯度为：

$$
\frac{\partial\mathcal{L}}{\partial\boldsymbol{Y}_{out}}=\frac{\partial\mathcal{L}}{\partial\boldsymbol{Z}}\boldsymbol{W}_{out}^{\top}
$$

由于 \(d_{out}\) 是常数，所以这个矩阵乘法不会额外引入关于 \(d\) 的求和增长。因此：

$$
\mathrm{RMS}\left(\frac{\partial\mathcal{L}}{\partial\boldsymbol{Y}_{out}}\right)=\Theta(d^{-\alpha})
$$

中间层梯度继承这个尺度：

$$
\mathrm{RMS}\left(\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}_k}\right)=\Theta(d^{-\alpha})
$$

由于：

$$
\boldsymbol{W}_k\in\mathbb{R}^{d\times d}
$$

所以：

$$
\left\|\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}_k}\right\|_F^2=d^2\cdot\Theta(d^{-\alpha})^2
$$

也就是：

$$
\left\|\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}_k}\right\|_F^2=\Theta(d^{2-2\alpha})
$$

损失增量为：

$$
\Delta\mathcal{L}_k\approx-\eta_k\left\|\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}_k}\right\|_F^2
$$

为了让：

$$
\Delta\mathcal{L}_k=\Theta(1)
$$

需要：

$$
\eta_k\cdot\Theta(d^{2-2\alpha})=\Theta(1)
$$

因此：

$$
\eta_k=\Theta(d^{2\alpha-2})
$$

参数更新量为：

$$
\Delta\boldsymbol{W}_k=-\eta_k\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}_k}
$$

所以：

$$
\mathrm{RMS}(\Delta\boldsymbol{W}_k)=\eta_k\cdot\mathrm{RMS}\left(\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}_k}\right)
$$

代入前面的两个尺度：

$$
\mathrm{RMS}(\Delta\boldsymbol{W}_k)=\Theta(d^{2\alpha-2})\cdot\Theta(d^{-\alpha})
$$

得到：

$$
\mathrm{RMS}(\Delta\boldsymbol{W}_k)=\Theta(d^{\alpha-2})
$$

而特征变化条件要求：

$$
\mathrm{RMS}(\Delta\boldsymbol{W}_k)=\Theta\left(\frac{1}{d}\right)
$$

也就是：

$$
\mathrm{RMS}(\Delta\boldsymbol{W}_k)=\Theta(d^{-1})
$$

因此：

$$
\Theta(d^{\alpha-2})=\Theta(d^{-1})
$$

比较指数：

$$
\alpha-2=-1
$$

所以：

$$
\alpha=1
$$

于是：

$$
\mathrm{RMS}(\boldsymbol{W}_{out})=\Theta\left(\frac{1}{d}\right)
$$

最终得到：

$$
\mathrm{Var}(\boldsymbol{W}_{out})=\Theta\left(\frac{1}{d^2}\right)
$$

这就解释了为什么前面要把输出层初始化方差设为：

$$
\mathrm{Var}(\boldsymbol{W}_{out})\propto\frac{1}{d^2}
$$

单看 \(\Delta\mathcal{L}=\Theta(1)\) 无法唯一推出这个设定；只有再加入特征变化条件：

$$
\mathrm{RMS}(\Delta\boldsymbol{Y}_k)=\Theta(1)
$$

才能排除其他选择，并推出 MuP 所需的输出层初始化尺度。



## 参考资料

1. Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466). NeurIPS 2021. [PDF](https://arxiv.org/pdf/2203.03466)
2. 苏剑林. [《初探MuP：超参数的跨模型尺度迁移规律》](https://spaces.ac.cn/archives/10770). 科学空间, 2025-03-13.
3. 苏剑林. [《高阶MuP：更简明但更高明的谱条件缩放》](https://spaces.ac.cn/archives/10795). 科学空间, 2025-03-24.
4. Microsoft Research. [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://www.microsoft.com/en-us/research/publication/tuning-large-neural-networks-via-zero-shot-hyperparameter-transfer/)
5. Microsoft. [microsoft/mup: Maximal Update Parametrization and Hyperparameter Transfer](https://github.com/microsoft/mup)
