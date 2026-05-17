---
date: '2026-05-17T10:15:54+08:00'
draft: false
title: 'LLM Theory 03: 权重更新的拆解——SMD'
categories: ["LLM Theory"]
tags: ["LLM", "LLM Theory", "Optimization", "SMD", "Normalization", "Weight Decay", "SGD"]
series: ["LLM Theory"]
series_order: 3
weight: 3
math: true
---

> 本文是 LLM Theory 系列中关于 Spherical Motion Dynamics（SMD）的学习笔记，主要参考 Wan et al. 的
> [Spherical Motion Dynamics: Learning Dynamics of Neural Network with Normalization, Weight Decay, and SGD](https://arxiv.org/abs/2006.08419)
> 以及 NeurIPS 2021 版本
> [Spherical Motion Dynamics: Learning Dynamics of Normalized Neural Network using SGD and Weight Decay](https://proceedings.neurips.cc/paper/2021/hash/326a8c055c0d04f5b06544665d8bb3ea-Abstract.html)。

## 朴素的描述模型更新

对于本科一二年级时候的笔者而言，如果想要描述模型的更新量，那么只会考虑这个非常直接的东西：$\lVert \boldsymbol{W}_{t+1} - \boldsymbol{W}_{t} \rVert$。
这确实是很直白的，相当于计算了模型权重的欧氏距离。是什么让我们必须放弃这种非常直观的欧氏距离呢？源于常用的归一化。
加上归一化之后，对于模型的输出，$y = \operatorname{BN}(x, k\boldsymbol{W}) = \operatorname{BN}(x, \boldsymbol{W})$，看不出模型尺度对 $y$ 的影响。但是却会从 $\lVert k\boldsymbol{W}_{t+1} - k\boldsymbol{W}_{t} \rVert$ 这一度量手段上产生 $k$ 倍的差距。
这就发现一个很显然的问题了：我们是希望通过观测类似于 $\lVert \boldsymbol{W}_{t+1} - \boldsymbol{W}_{t} \rVert$ 这样的东西来控制模型的训练，而不是直接看最后的 $y$。因为只通过看 $y$ 的变化，并基于这种变化的规律，去指导如何进行训练前的设置，这件事并不容易。我们是希望找到一个更可以写出明确表达式，更容易观测，意义更明确的指标去观测，而这个指标恰好还要有一些规律和 $y$ 的规律“趋同”，这样我们就得到了一个 $y$ 的近似物，而且这个近似物比 $y$ 更好分析。所以我们会很直接的想到 $\lVert \boldsymbol{W}_{t+1} - \boldsymbol{W}_{t} \rVert$。但是现在，当我们给模型加上BN（为什么加BN不赘述了）以后发现的问题是 $\lVert \boldsymbol{W}_{t+1} - \boldsymbol{W}_{t} \rVert$ 会随着模型尺度产生成倍的差距，而 $y$ 却不会被尺度影响。这说明 $\lVert \boldsymbol{W}_{t+1} - \boldsymbol{W}_{t} \rVert$ 不是一个好的指标，因为它错误地预测了 $y$ 随着尺度变化的行为规律。
当然，原则上，我们其实可以发现，上面的地方有一个逻辑漏洞：为什么权重变化一定能预测 $y$ 的变化，这是错误的呀，权重变了 $y$ 不一定变。所以 $\lVert \boldsymbol{W}_{t+1} - \boldsymbol{W}_{t} \rVert$ 这个东西失效是情理之中的，不能先入为主的认为 $\lVert \boldsymbol{W}_{t+1} - \boldsymbol{W}_{t} \rVert$ 的规律能和 $y$ 趋同，那么出现上面的问题十分合理。我们一个理想中的目标是通过权重变化去预测 $y$ 的变化，这样我们就可以通过设计一套权重变化的方案去精确得到想要的 $y$。理想是很远大的，现实会比较难做，但是可以在最朴素的方案上向前不断推进预测的细化程度。SMD就是这样一种东西。

## 更可解释的模型更新--SMD方法论

笔者决定先把基于SMD分析模型更新量的方案给出，然后说明这种方案的合理性。
如果某一层权重是尺度无关的，比如BN层。记作 $\boldsymbol{W}$。先把 $\boldsymbol{W}$ 写成 $\boldsymbol{W}_t = r_t \boldsymbol{U}_t$，$\boldsymbol{U}_t$ 是单位的权重。

{{< math >}}
r_t = \lVert \boldsymbol{W}_t \rVert
{{< /math >}}

设优化器给出的更新为：

{{< math >}}
\boldsymbol{W}_{t+1} = \boldsymbol{W}_t + \boldsymbol{\delta}_t
{{< /math >}}

这个时候要对 $\boldsymbol{\delta}_t$ 做一次关于 $\boldsymbol{W}_t$ 方向的分解：

{{< math >}}
\boldsymbol{\delta}_t = \boldsymbol{\delta}_r + \boldsymbol{\delta}_u
{{< /math >}}

其中 $\boldsymbol{\delta}_r$ 是 $\boldsymbol{W}_t$ 的径向方向更新，$\boldsymbol{\delta}_u$ 是 $\boldsymbol{W}_t$ 的切向方向更新：

{{< math >}}
\boldsymbol{\delta}_r = \left\langle \boldsymbol{\delta}_t, \boldsymbol{U}_t \right\rangle \boldsymbol{U}_t
{{< /math >}}

那么我们主要分析的其实是两点，一点是 $\boldsymbol{\delta}_r$ 要稳定，另一点是 $\boldsymbol{\delta}_u$ 是否真的有改变。
因为 $L(k\boldsymbol{W}) = L(\boldsymbol{W})$（源于尺度无关的假设），而 $k\boldsymbol{W}$ 相对 $\boldsymbol{W}$ 也只是在 $\boldsymbol{W}$ 的切向做了变换，这说明在尺度无关模型模型改变量只是 $\boldsymbol{W}$ 径向的改变量。相较于之前的欧氏距离，我们实际上对改变量这个事情进行了细化，细化到了 $\boldsymbol{W}$ 的径向方向。

## 用SMD分析朴素的SGD
注意，所有分析都有一个前提，就是这个层加了BN或者其他能让尺度无关现象发生的操作。
对于朴素 SGD：

{{< math >}}
\boldsymbol{W}_{t+1} = \boldsymbol{W}_{t} - \eta \boldsymbol{g}_{t}
{{< /math >}}


由于 normalization 导致：

{{< math >}}
\boldsymbol{g}_{t} \perp \boldsymbol{W}_{t}
{{< /math >}}


所以：

{{< math >}}
\boldsymbol{\delta}_t = -\eta \boldsymbol{g}_{t}
{{< /math >}}


是纯切向更新。

因此：

{{< math >}}
\boldsymbol{\delta}_{r} = \boldsymbol{0}, \qquad \boldsymbol{\delta}_{u} = -\eta \boldsymbol{g}_{t}
{{< /math >}}


于是：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\left\lVert \boldsymbol{\delta}_{u} \right\rVert^2
{{< /math >}}

也就是：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\eta^2 \left\lVert \boldsymbol{g}_{t} \right\rVert^2
{{< /math >}}


朴素 SGD 没有径向收缩项 $\boldsymbol{\delta}_{r}$，只有切向更新 $\boldsymbol{\delta}_{u}$。但切向更新虽然改变方向，却会二阶地增加半径。

角更新则是：

{{< math >}}
\Delta_t
\approx
\frac{\left\lVert \boldsymbol{\delta}_{u} \right\rVert}
{\left\lVert \boldsymbol{W}_{t} \right\rVert}
{{< /math >}}


对于朴素 SGD：

{{< math >}}
\Delta_t
\approx
\frac{\eta \left\lVert \boldsymbol{g}_{t} \right\rVert}
{\left\lVert \boldsymbol{W}_{t} \right\rVert}
{{< /math >}}


再用 scale-invariant gradient：

{{< math >}}
\boldsymbol{g}_{t}
=
\frac{1}{\left\lVert \boldsymbol{W}_{t} \right\rVert}
\tilde{\boldsymbol{g}}_{t}
{{< /math >}}


得到：

{{< math >}}
\Delta_t
\approx
\frac{\eta \left\lVert \tilde{\boldsymbol{g}}_{t} \right\rVert}
{\left\lVert \boldsymbol{W}_{t} \right\rVert^2}
{{< /math >}}


## 用SMD分析SGD with Weight Decay

有了朴素 SGD 的铺垫之后，SGD + Weight Decay 就很好拆了。

先写更新式：

{{< math >}}
\boldsymbol{W}_{t+1}
=
\boldsymbol{W}_{t}
-
\eta\left(
\boldsymbol{g}_{t}
+
\lambda \boldsymbol{W}_{t}
\right)
{{< /math >}}

这里的 $\boldsymbol{g}_{t}$ 是任务 loss 对 $\boldsymbol{W}_t$ 的梯度，先不把 WD 算进 loss 里面。展开之后就是：

{{< math >}}
\boldsymbol{W}_{t+1}
=
\boldsymbol{W}_{t}
-
\eta\boldsymbol{g}_{t}
-
\eta\lambda\boldsymbol{W}_{t}
{{< /math >}}

所以总更新量是：

{{< math >}}
\boldsymbol{\delta}_t
=
\boldsymbol{W}_{t+1}
-
\boldsymbol{W}_{t}
=
-
\eta\boldsymbol{g}_{t}
-
\eta\lambda\boldsymbol{W}_{t}
{{< /math >}}

因为尺度无关带来的正交性质：

{{< math >}}
\left\langle \boldsymbol{W}_{t}, \boldsymbol{g}_{t} \right\rangle = 0
{{< /math >}}

所以 $\boldsymbol{g}_t$ 还是切向的。这个时候 WD 项就很有意思了，$-\eta\lambda\boldsymbol{W}_{t}$ 和 $\boldsymbol{W}_{t}$ 平行，因此它不是切向更新，而是一个非常干净的径向收缩。

所以按照前面的记号，可以直接拆成：

{{< math >}}
\boldsymbol{\delta}_{r}
=
-
\eta\lambda\boldsymbol{W}_{t}
{{< /math >}}

{{< math >}}
\boldsymbol{\delta}_{u}
=
-
\eta\boldsymbol{g}_{t}
{{< /math >}}

也就是：

{{< math >}}
\boldsymbol{\delta}_{t}
=
\boldsymbol{\delta}_{r}
+
\boldsymbol{\delta}_{u}
{{< /math >}}

简单描述就是，$\boldsymbol{\delta}_r$ 是 WD 给出的径向收缩，把 $\boldsymbol{W}$ 往原点拉；$\boldsymbol{\delta}_u$ 是任务梯度给出的切向更新，推动 $\boldsymbol{W}$ 在球面上转方向。

接下来看半径怎么变化。由于：

{{< math >}}
\boldsymbol{W}_t + \boldsymbol{\delta}_r
=
\left(1-\eta\lambda\right)\boldsymbol{W}_t
{{< /math >}}

并且 $\boldsymbol{\delta}_u \perp \boldsymbol{W}_t$，所以交叉项为 0：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left\lVert
\left(1-\eta\lambda\right)\boldsymbol{W}_t
+
\boldsymbol{\delta}_u
\right\rVert^2
{{< /math >}}

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left(1-\eta\lambda\right)^2
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\left\lVert \boldsymbol{\delta}_{u} \right\rVert^2
{{< /math >}}

代入 $\boldsymbol{\delta}_{u} = -\eta\boldsymbol{g}_t$：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left(1-\eta\lambda\right)^2
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\eta^2
\left\lVert \boldsymbol{g}_{t} \right\rVert^2
{{< /math >}}

这里和朴素 SGD 的区别就很清楚了。朴素 SGD 没有 $\boldsymbol{\delta}_r$，所以半径只会因为切向更新的二阶项变大；SGD + WD 多了一个 $-\eta\lambda\boldsymbol{W}_t$，它会把半径往回压。于是训练过程里同时存在两个方向相反的东西：WD 带来的径向收缩，和切向运动带来的离心增长。

再代入 scale-invariant gradient：

{{< math >}}
\boldsymbol{g}_{t}
=
\frac{1}{\left\lVert \boldsymbol{W}_{t} \right\rVert}
\tilde{\boldsymbol{g}}_{t}
{{< /math >}}

可以得到：

{{< math >}}
\left\lVert \boldsymbol{g}_{t} \right\rVert^2
=
\frac{
\left\lVert \tilde{\boldsymbol{g}}_{t} \right\rVert^2
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
}
{{< /math >}}

所以半径动力学可以写成：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left(1-\eta\lambda\right)^2
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\frac{
\eta^2
\left\lVert \tilde{\boldsymbol{g}}_{t} \right\rVert^2
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
}
{{< /math >}}

这个式子其实已经把 SMD 的图像写出来了。当 $\lVert \boldsymbol{W}_t \rVert$ 很大时，WD 的收缩很明显，而切向梯度项因为分母很大反而不明显；当 $\lVert \boldsymbol{W}_t \rVert$ 很小时，切向梯度带来的离心增长会变强。所以 SGD + WD 不再像朴素 SGD 那样让半径一直长大，而是有机会稳定到某个平衡值。

如果用小步长近似去看半径本身的变化，可以写成：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert
-
\left\lVert \boldsymbol{W}_{t} \right\rVert
\approx
-
\eta\lambda
\left\lVert \boldsymbol{W}_{t} \right\rVert
+
\frac{
\eta^2
\left\lVert \tilde{\boldsymbol{g}}_{t} \right\rVert^2
}{
2
\left\lVert \boldsymbol{W}_{t} \right\rVert^3
}
{{< /math >}}

前一项是 WD 的向心项，后一项是切向更新带来的离心项。平衡的时候可以认为：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert
\approx
\left\lVert \boldsymbol{W}_{t} \right\rVert
{{< /math >}}

所以：

{{< math >}}
-
\eta\lambda
\left\lVert \boldsymbol{W}_{t} \right\rVert
+
\frac{
\eta^2
\left\lVert \tilde{\boldsymbol{g}}_{t} \right\rVert^2
}{
2
\left\lVert \boldsymbol{W}_{t} \right\rVert^3
}
\approx
0
{{< /math >}}

移项之后得到：

{{< math >}}
\left\lVert \boldsymbol{W}_{t} \right\rVert^4
\approx
\frac{
\eta
\left\lVert \tilde{\boldsymbol{g}}_{t} \right\rVert^2
}{
2\lambda
}
{{< /math >}}

如果令：

{{< math >}}
L
=
\mathbb{E}
\left[
\left\lVert \tilde{\boldsymbol{g}}_{t} \right\rVert^2
\right]
{{< /math >}}

那么平衡半径就是：

{{< math >}}
\left\lVert \boldsymbol{W} \right\rVert^*
=
\left(
\frac{L\eta}{2\lambda}
\right)^{1/4}
{{< /math >}}

这个结论也解释了为什么在有 normalization 的网络里，WD 不能只被理解成“普通正则项”。它实际上在控制 scale-invariant weight 的半径，并且和切向梯度共同决定了半径的 equilibrium。

最后看角更新。角更新主要由切向更新决定：

{{< math >}}
\Delta_t
\approx
\frac{
\left\lVert \boldsymbol{\delta}_{u} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert
}
{{< /math >}}

SGD + WD 里面 $\boldsymbol{\delta}_{u} = -\eta\boldsymbol{g}_t$，所以：

{{< math >}}
\Delta_t
\approx
\frac{
\eta
\left\lVert \boldsymbol{g}_{t} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert
}
=
\frac{
\eta
\left\lVert \tilde{\boldsymbol{g}}_{t} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
}
{{< /math >}}

把平衡半径代回去：

{{< math >}}
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
\approx
\left\lVert \tilde{\boldsymbol{g}}_{t} \right\rVert
\sqrt{
\frac{\eta}{2\lambda}
}
{{< /math >}}

因此平衡时：

{{< math >}}
\Delta^*
\approx
\sqrt{2\eta\lambda}
{{< /math >}}

这就很有意思了：进入 equilibrium 以后，角更新不再显式依赖 $\lVert \tilde{\boldsymbol{g}}_t \rVert$，而主要由学习率 $\eta$ 和 weight decay $\lambda$ 决定。也就是说，SGD + WD 的 SMD 意义不是“SGD 多加了一个惩罚项”，而是 WD 给 scale-invariant weight 加了一个径向控制器，让朴素 SGD 的半径增长变成稳定半径上的球面运动。

## 用SMD分析Adam


同样，这一层权重是 scale-invariant 的，也就是：

{{< math >}}
L(\alpha\boldsymbol{W}) = L(\boldsymbol{W})
{{< /math >}}

因此任务梯度满足：

{{< math >}}
\left\langle \boldsymbol{W}_{t}, \boldsymbol{g}_{t} \right\rangle = 0
{{< /math >}}

这里：

{{< math >}}
\boldsymbol{g}_{t}
=
\nabla_{\boldsymbol{W}} L(\boldsymbol{W}_{t})
{{< /math >}}

也就是说，raw gradient $\boldsymbol{g}_t$ 是切向的。但是 更新权重的方向里面加了动量之类的东西，比较复杂。

先忽略 bias correction，Adam 的更新式是：

{{< math >}}
\boldsymbol{m}_{t}
=
\beta_1 \boldsymbol{m}_{t-1}
+
\left(1-\beta_1\right)\boldsymbol{g}_{t}
{{< /math >}}

{{< math >}}
\boldsymbol{v}_{t}
=
\beta_2 \boldsymbol{v}_{t-1}
+
\left(1-\beta_2\right)\boldsymbol{g}_{t}^{2}
{{< /math >}}

{{< math >}}
\boldsymbol{q}_{t}
=
\frac{
\boldsymbol{m}_{t}
}{
\sqrt{\boldsymbol{v}_{t}}+\epsilon
}
{{< /math >}}

{{< math >}}
\boldsymbol{W}_{t+1}
=
\boldsymbol{W}_{t}
-
\eta\boldsymbol{q}_{t}
{{< /math >}}

所以总更新量是：

{{< math >}}
\boldsymbol{\delta}_{t}
=
\boldsymbol{W}_{t+1}
-
\boldsymbol{W}_{t}
=
-
\eta\boldsymbol{q}_{t}
{{< /math >}}

现在还是按前面的方式，把 $\boldsymbol{\delta}_t$ 拆成径向更新和切向更新：

{{< math >}}
\boldsymbol{\delta}_{t}
=
\boldsymbol{\delta}_{r}
+
\boldsymbol{\delta}_{u}
{{< /math >}}

径向分量就是 $\boldsymbol{\delta}_t$ 在 $\boldsymbol{W}_t$ 方向上的投影：

{{< math >}}
\boldsymbol{\delta}_{r}
=
\frac{
\left\langle \boldsymbol{\delta}_{t}, \boldsymbol{W}_{t} \right\rangle
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
}
\boldsymbol{W}_{t}
{{< /math >}}

代入 $\boldsymbol{\delta}_t = -\eta\boldsymbol{q}_t$：

{{< math >}}
\boldsymbol{\delta}_{r}
=
-
\eta
\frac{
\left\langle \boldsymbol{q}_{t}, \boldsymbol{W}_{t} \right\rangle
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
}
\boldsymbol{W}_{t}
{{< /math >}}

令：

{{< math >}}
c_t
=
\frac{
\left\langle \boldsymbol{q}_{t}, \boldsymbol{W}_{t} \right\rangle
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
}
{{< /math >}}

那么：

{{< math >}}
\boldsymbol{\delta}_{r}
=
-
\eta c_t \boldsymbol{W}_{t}
{{< /math >}}

切向分量就是剩下的部分：

{{< math >}}
\boldsymbol{\delta}_{u}
=
\boldsymbol{\delta}_{t}
-
\boldsymbol{\delta}_{r}
=
-
\eta
\left(
\boldsymbol{q}_{t}
-
c_t\boldsymbol{W}_{t}
\right)
{{< /math >}}

令：

{{< math >}}
\boldsymbol{q}_{u,t}
=
\boldsymbol{q}_{t}
-
c_t\boldsymbol{W}_{t}
{{< /math >}}

那么有：

{{< math >}}
\boldsymbol{q}_{u,t} \perp \boldsymbol{W}_{t}
{{< /math >}}

因此：

{{< math >}}
\boldsymbol{\delta}_{u}
=
-
\eta\boldsymbol{q}_{u,t}
{{< /math >}}

这里就是 Adam 和朴素 SGD 最关键的差别。对于朴素 SGD，$\boldsymbol{\delta}_t = -\eta\boldsymbol{g}_t$，而 $\boldsymbol{g}_t \perp \boldsymbol{W}_t$，所以它是纯切向更新，$\boldsymbol{\delta}_r = \boldsymbol{0}$。

但是 Adam 不一样。虽然 raw gradient $\boldsymbol{g}_t$ 是切向的，Adam 实际使用的却是：

{{< math >}}
\boldsymbol{q}_{t}
=
\frac{
\boldsymbol{m}_{t}
}{
\sqrt{\boldsymbol{v}_{t}}+\epsilon
}
{{< /math >}}

动量项和逐元素缩放一般不会保持“和 $\boldsymbol{W}_t$ 正交”这个性质。也就是说：

{{< math >}}
\left\langle \boldsymbol{g}_{t}, \boldsymbol{W}_{t} \right\rangle = 0
{{< /math >}}

并不能推出：

{{< math >}}
\left\langle \boldsymbol{q}_{t}, \boldsymbol{W}_{t} \right\rangle = 0
{{< /math >}}

所以 Adam 的更新通常会有：

{{< math >}}
c_t \neq 0
{{< /math >}}

Adam 会把原本切向的 normalized gradient，经过动量和逐元素预条件之后，变成一个既有切向分量、也可能有径向分量的更新。（真坏啊）

接着看径向动力学。因为：

{{< math >}}
\boldsymbol{W}_t + \boldsymbol{\delta}_{r}
=
\left(1-\eta c_t\right)\boldsymbol{W}_t
{{< /math >}}

同时 $\boldsymbol{\delta}_u \perp \boldsymbol{W}_t$，所以：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left(1-\eta c_t\right)^2
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\left\lVert \boldsymbol{\delta}_{u} \right\rVert^2
{{< /math >}}

代入 $\boldsymbol{\delta}_u = -\eta\boldsymbol{q}_{u,t}$：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left(1-\eta c_t\right)^2
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\eta^2
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
{{< /math >}}

小步长近似下：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
-
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
\approx
-
2\eta c_t
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\eta^2
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
{{< /math >}}

这里依然可以看到两部分。第一项来自 $\boldsymbol{\delta}_r$，也就是 Adam 自己产生的径向分量；第二项来自 $\boldsymbol{\delta}_u$，也就是切向更新带来的离心增长。

这个时候 $c_t$ 的符号就很重要了。如果 $c_t > 0$，那么 $\boldsymbol{\delta}_r = -\eta c_t\boldsymbol{W}_t$ 是向内收缩，半径倾向于变小。如果 $c_t < 0$，那么 $\boldsymbol{\delta}_r$ 和 $\boldsymbol{W}_t$ 同向，是向外扩张，半径倾向于变大。如果 $c_t \approx 0$，Adam 近似没有径向力，半径主要由切向离心项撑大。

所以 Adam 的径向动力学被这个量控制：

{{< math >}}
c_t
=
\frac{
\left\langle \boldsymbol{q}_{t}, \boldsymbol{W}_{t} \right\rangle
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
}
{{< /math >}}

它被 moment、二阶矩估计和element-wise缩放共同作用。因此 Adam 的径向力更像是adaptive preconditioner 带来的。

再看角更新。角更新主要由切向分量决定：

{{< math >}}
\Delta_t
\approx
\frac{
\left\lVert \boldsymbol{\delta}_{u} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert
}
{{< /math >}}

Adam 中：

{{< math >}}
\boldsymbol{\delta}_{u}
=
-
\eta\boldsymbol{q}_{u,t}
{{< /math >}}

所以：

{{< math >}}
\Delta_t
\approx
\frac{
\eta
\left\lVert \boldsymbol{q}_{u,t} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert
}
{{< /math >}}

这也和 SGD 不一样。SGD 里面因为 scale-invariant gradient 会带来 $\lVert \boldsymbol{g}_t \rVert \propto 1/\lVert \boldsymbol{W}_t \rVert$，所以角更新大概按 $1/\lVert \boldsymbol{W}_t \rVert^2$ 衰减。

Adam 的 $\boldsymbol{q}_t$ 做了类似梯度尺度归一化的操作。如果 $\boldsymbol{g}_t$ 因为权重尺度变大而按 $1/\alpha$ 缩小，那么 $\boldsymbol{m}_t$ 大致也按 $1/\alpha$ 缩小，$\boldsymbol{v}_t$ 大致按 $1/\alpha^2$ 缩小，于是：

{{< math >}}
\boldsymbol{q}_t
=
\frac{
\boldsymbol{m}_{t}
}{
\sqrt{\boldsymbol{v}_{t}}+\epsilon
}
{{< /math >}}

在 $\epsilon$ 不主导的时候，$\boldsymbol{q}_t$ 对这种整体尺度变化并不敏感。因此 Adam 的角更新更像是：

{{< math >}}
\Delta_t^{\mathrm{Adam}}
\approx
\frac{
\eta
\left\lVert \boldsymbol{q}_{u,t} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert
}
{{< /math >}}

也就是更接近按 $1/\lVert \boldsymbol{W}_t \rVert$ 衰减，而不是 SGD 那种 $1/\lVert \boldsymbol{W}_t \rVert^2$ 衰减。当然这里有一个前提，就是 $\epsilon$ 没有主导分母；如果 $\epsilon$ 主导了，那这个尺度抵消就会变弱。

最后说 equilibrium。朴素 Adam 没有一个很干净的平衡半径公式，因为它的径向项不是固定超参数给出来的，而是：

{{< math >}}
\boldsymbol{\delta}_{r}
=
-
\eta c_t\boldsymbol{W}_{t}
{{< /math >}}

其中 $c_t$ 可能为正，可能为负，也可能随着训练变化。因此 Adam 的平衡条件最多只能先写成形式上的：

{{< math >}}
-
2\eta c_t
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\eta^2
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
\approx
0
{{< /math >}}

也就是：

{{< math >}}
2c_t
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
\approx
\eta
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
{{< /math >}}

如果额外假设 $c_t > 0$，并且 $c_t$ 和 $\lVert \boldsymbol{q}_{u,t} \rVert^2$ 在局部都比较稳定，那么可以形式上写成：

{{< math >}}
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
\approx
\frac{
\eta
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
}{
2c_t
}
{{< /math >}}

所以关键就是径向力c要和adam update的径向力差不多能抵消，这个没有太确切的解法，感觉应该需要调参。

## 用SMD分析AdamW

AdamW 可以看成是在 Adam 的自适应更新之外，又显式加了一个 decoupled weight decay。这里依然只分析带 normalization 的 scale-invariant weight：

{{< math >}}
L(\alpha\boldsymbol{W}) = L(\boldsymbol{W})
{{< /math >}}

因此原始任务梯度还是满足：

{{< math >}}
\left\langle \boldsymbol{W}_{t}, \boldsymbol{g}_{t} \right\rangle = 0
{{< /math >}}

AdamW 的 moment 部分和 Adam 一样：

{{< math >}}
\boldsymbol{m}_{t}
=
\beta_1 \boldsymbol{m}_{t-1}
+
\left(1-\beta_1\right)\boldsymbol{g}_{t}
{{< /math >}}

{{< math >}}
\boldsymbol{v}_{t}
=
\beta_2 \boldsymbol{v}_{t-1}
+
\left(1-\beta_2\right)\boldsymbol{g}_{t}^{2}
{{< /math >}}

{{< math >}}
\boldsymbol{q}_{t}
=
\frac{
\boldsymbol{m}_{t}
}{
\sqrt{\boldsymbol{v}_{t}}+\epsilon
}
{{< /math >}}

区别在更新式：

{{< math >}}
\boldsymbol{W}_{t+1}
=
\boldsymbol{W}_{t}
-
\eta\boldsymbol{q}_{t}
-
\eta\lambda\boldsymbol{W}_{t}
{{< /math >}}

其中 $-\eta\boldsymbol{q}_t$ 是自适应梯度更新，$-\eta\lambda\boldsymbol{W}_t$ 是 decoupled weight decay。总更新量为：

{{< math >}}
\boldsymbol{\delta}_{t}
=
\boldsymbol{W}_{t+1}
-
\boldsymbol{W}_{t}
=
-
\eta\boldsymbol{q}_{t}
-
\eta\lambda\boldsymbol{W}_{t}
{{< /math >}}

和 Adam 一样，虽然 $\boldsymbol{g}_t \perp \boldsymbol{W}_t$，但是 $\boldsymbol{q}_t$ 不一定还和 $\boldsymbol{W}_t$ 垂直。所以先把 $\boldsymbol{q}_t$ 分解成径向和切向两部分。

定义：

{{< math >}}
c_t
=
\frac{
\left\langle \boldsymbol{q}_{t}, \boldsymbol{W}_{t} \right\rangle
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
}
{{< /math >}}

于是：

{{< math >}}
\boldsymbol{q}_{t}
=
c_t\boldsymbol{W}_{t}
+
\boldsymbol{q}_{u,t}
{{< /math >}}

其中：

{{< math >}}
\boldsymbol{q}_{u,t}
=
\boldsymbol{q}_{t}
-
c_t\boldsymbol{W}_{t}
{{< /math >}}

并且：

{{< math >}}
\boldsymbol{q}_{u,t} \perp \boldsymbol{W}_{t}
{{< /math >}}

这里 $c_t\boldsymbol{W}_t$ 是 AdamW update 里的径向改变，$\boldsymbol{q}_{u,t}$ 是切向改变。

代回 AdamW 的总更新：

{{< math >}}
\boldsymbol{\delta}_{t}
=
-
\eta
\left(
c_t\boldsymbol{W}_{t}
+
\boldsymbol{q}_{u,t}
\right)
-
\eta\lambda\boldsymbol{W}_{t}
{{< /math >}}

整理一下：

{{< math >}}
\boldsymbol{\delta}_{t}
=
-
\eta
\left(
\lambda+c_t
\right)
\boldsymbol{W}_{t}
-
\eta\boldsymbol{q}_{u,t}
{{< /math >}}

所以 AdamW 的 SMD 分解就是：

{{< math >}}
\boldsymbol{\delta}_{r}
=
-
\eta
\left(
\lambda+c_t
\right)
\boldsymbol{W}_{t}
{{< /math >}}

{{< math >}}
\boldsymbol{\delta}_{u}
=
-
\eta\boldsymbol{q}_{u,t}
{{< /math >}}

这里很清楚：$-\eta\lambda\boldsymbol{W}_t$ 是显式 weight decay 带来的径向收缩，$-\eta c_t\boldsymbol{W}_t$ 是 AdamW 自适应更新自己产生的径向分量，$-\eta\boldsymbol{q}_{u,t}$ 是切向更新，主要改变权重方向。

接下来看半径动力学。由：

{{< math >}}
\boldsymbol{W}_{t}
+
\boldsymbol{\delta}_{r}
=
\left(
1-\eta\left(\lambda+c_t\right)
\right)
\boldsymbol{W}_{t}
{{< /math >}}

又因为 $\boldsymbol{\delta}_{u} \perp \boldsymbol{W}_{t}$，所以：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left(
1-\eta\left(\lambda+c_t\right)
\right)^2
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\left\lVert \boldsymbol{\delta}_{u} \right\rVert^2
{{< /math >}}

代入 $\boldsymbol{\delta}_{u} = -\eta\boldsymbol{q}_{u,t}$：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
=
\left(
1-\eta\left(\lambda+c_t\right)
\right)^2
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\eta^2
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
{{< /math >}}

小步长近似下：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
-
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
\approx
-
2\eta
\left(
\lambda+c_t
\right)
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\eta^2
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
{{< /math >}}

这个式子就是 AdamW 的半径动力学。前一项是径向项，后一项是切向更新带来的离心增长。

所以 AdamW 里面真正控制径向行为的是：

{{< math >}}
\lambda+c_t
{{< /math >}}

$\lambda$ 是显式 weight decay，通常提供向内收缩；$c_t$ 来自自适应更新 $\boldsymbol{q}_t$ 对 $\boldsymbol{W}_t$ 的径向投影。如果 $\lambda+c_t > 0$，那么 $\boldsymbol{\delta}_r$ 是向内的，半径倾向于减小。如果 $\lambda+c_t < 0$，那么 $\boldsymbol{\delta}_r$ 是向外的，半径倾向于增大。如果 $\lambda+c_t \approx 0$，那么显式 weight decay 和自适应更新的径向改变大致抵消，半径主要由切向离心项影响。

所以不难看出AdamW 相对于Adam的优势是Adam 只有隐式的 $c_t$，而 AdamW 至少多了一个显式可控的 $\lambda$。但是要注意，AdamW 不是只有 $\lambda$ 这一个径向项，因为 $\boldsymbol{q}_t$ 仍然可能通过 $c_t$ 引入额外径向改变。这里提一个思考，SMD分解的理论完全对吗？AdamW已经被广泛的用于LLM训练，如果存在问题，应该不会一直沿用。

角更新还是主要由切向分量决定：

{{< math >}}
\Delta_t
\approx
\frac{
\left\lVert \boldsymbol{\delta}_{u} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert
}
{{< /math >}}

AdamW 中：

{{< math >}}
\boldsymbol{\delta}_{u}
=
-
\eta\boldsymbol{q}_{u,t}
{{< /math >}}

所以：

{{< math >}}
\Delta_t
\approx
\frac{
\eta
\left\lVert \boldsymbol{q}_{u,t} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert
}
{{< /math >}}

如果 $\boldsymbol{q}_t$ 主要是切向的，也就是 $c_t \approx 0$，那么 $\boldsymbol{q}_{u,t} \approx \boldsymbol{q}_t$，此时：

{{< math >}}
\Delta_t
\approx
\frac{
\eta
\left\lVert \boldsymbol{q}_{t} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert
}
{{< /math >}}

最后写一下形式上的平衡条件。半径平衡要求：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert
\approx
\left\lVert \boldsymbol{W}_{t} \right\rVert
{{< /math >}}

也就是：

{{< math >}}
-
2\eta
\left(
\lambda+c_t
\right)
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\eta^2
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
\approx
0
{{< /math >}}

约掉一个 $\eta$：

{{< math >}}
2
\left(
\lambda+c_t
\right)
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
\approx
\eta
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
{{< /math >}}

因此形式上的平衡半径满足：

{{< math >}}
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
\approx
\frac{
\eta
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
}{
2
\left(
\lambda+c_t
\right)
}
{{< /math >}}

也就是：

{{< math >}}
\left\lVert \boldsymbol{W}_{t} \right\rVert
\approx
\sqrt{
\frac{
\eta
\left\lVert \boldsymbol{q}_{u,t} \right\rVert^2
}{
2
\left(
\lambda+c_t
\right)
}
}
{{< /math >}}

这个式子成立的前提是 $\lambda+c_t > 0$，并且 $c_t$、$\lVert \boldsymbol{q}_{u,t} \rVert^2$ 在局部训练窗口内比较稳定。

如果考虑一个更理想的情况：AdamW 的自适应更新几乎没有径向改变，也就是 $c_t \approx 0$，那么：

{{< math >}}
\boldsymbol{\delta}_{r}
\approx
-
\eta\lambda\boldsymbol{W}_{t}
{{< /math >}}

{{< math >}}
\boldsymbol{\delta}_{u}
\approx
-
\eta\boldsymbol{q}_{t}
{{< /math >}}

半径动力学近似变成：

{{< math >}}
\left\lVert \boldsymbol{W}_{t+1} \right\rVert^2
\approx
\left(
1-\eta\lambda
\right)^2
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
+
\eta^2
\left\lVert \boldsymbol{q}_{t} \right\rVert^2
{{< /math >}}

平衡时：

{{< math >}}
\left\lVert \boldsymbol{W}_{t} \right\rVert^2
\approx
\frac{
\eta
\left\lVert \boldsymbol{q}_{t} \right\rVert^2
}{
2\lambda
}
{{< /math >}}

所以：

{{< math >}}
\left\lVert \boldsymbol{W}_{t} \right\rVert
\approx
\sqrt{
\frac{
\eta
\left\lVert \boldsymbol{q}_{t} \right\rVert^2
}{
2\lambda
}
}
{{< /math >}}

此时角更新为：

{{< math >}}
\Delta_t
\approx
\frac{
\eta
\left\lVert \boldsymbol{q}_{t} \right\rVert
}{
\left\lVert \boldsymbol{W}_{t} \right\rVert
}
{{< /math >}}

代入平衡半径，可以得到：

{{< math >}}
\Delta^*
\approx
\sqrt{2\eta\lambda}
{{< /math >}}

这个结果和前面的 SMD 直觉是相似的：如果 $c_t \approx 0$ 且 $\lVert \boldsymbol{q}_t \rVert$ 局部稳定，那么 AdamW 也会表现出稳定角更新。但是这个还是无法从理论上保证稳定。


## 用SMD分析Muon？

挖个坑以后再说。

## 参考文献

1. Ruosi Wan, Zhanxing Zhu, Xiangyu Zhang, Jian Sun. [Spherical Motion Dynamics: Learning Dynamics of Neural Network with Normalization, Weight Decay, and SGD](https://arxiv.org/abs/2006.08419). arXiv:2006.08419, 2020.
2. Ruosi Wan, Zhanxing Zhu, Xiangyu Zhang, Jian Sun. [Spherical Motion Dynamics: Learning Dynamics of Normalized Neural Network using SGD and Weight Decay](https://proceedings.neurips.cc/paper/2021/hash/326a8c055c0d04f5b06544665d8bb3ea-Abstract.html). NeurIPS 2021.
3. OpenReview: [Spherical Motion Dynamics: Learning Dynamics of Normalized Neural Network using SGD and Weight Decay](https://openreview.net/forum?id=RcbphT7qjTq).
