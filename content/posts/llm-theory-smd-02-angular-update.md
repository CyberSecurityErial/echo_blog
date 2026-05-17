---
date: '2026-05-17T17:56:11+08:00'
draft: true
title: 'LLM Theory: SMD 02 - Angular Update 才更接近真实改变量'
categories: ["LLM Theory"]
tags: ["LLM", "LLM Theory", "Optimization", "SMD", "Effective Learning Rate", "Angular Update", "Weight Decay"]
series: ["LLM Theory", "SMD"]
series_order: 2
weight: 4
math: true
---

> 本文是 LLM Theory 下 SMD 专题的第二篇。上一篇主要把 SGD、SGD + weight decay、Adam、AdamW 的径向/切向更新拆开了；这一篇继续沿着这个拆法，讨论几个笔者读 SMD 时觉得更关键的 insight。

## Insight 1：为什么只看 Effective Learning Rate 不够

在讨论 normalization 和 weight decay 的联合作用时，有一句非常关键的话：

> 对于 scale-invariant weight，任务梯度 $\partial L / \partial \boldsymbol{w}$ 总是与权重 $\boldsymbol{w}$ 垂直。因此，梯度分量总是倾向于增大权重范数，而 weight decay 提供的分量总是倾向于减小权重范数。

这句话其实就是 SMD 的基本物理图像：

{{< math >}}
\text{任务梯度}
\Rightarrow
\text{切向更新，改变方向，同时二阶增大范数}
{{< /math >}}

{{< math >}}
\text{Weight Decay}
\Rightarrow
\text{径向更新，缩小范数}
{{< /math >}}

也就是说，在带 normalization 的网络中，SGD 梯度并不是在普通欧氏空间里随便走，而是在球面的切线方向上推动权重转动；weight decay 则像一个向心力，把权重往原点拉。两者共同作用，最终可能让 weight norm 进入一个稳定状态。

但是这里有一个很容易被忽略的问题：

> Weight decay 可以对冲掉梯度更新导致的 weight norm 增长，但这并不意味着 gradient norm 本身也被稳定住了。

换句话说，weight norm 稳定，不代表整个训练动力学就稳定。

这个现象在一篇 weight decay 的博客里也能看到：加 weight decay 之后，weight norm 会逐渐减小并趋于稳定，但 grad norm 可能出现抖动甚至上升。参考 [Fantastic Pretraining Optimizers and Where to Find Them 2.2: The Hitchhiker's Guide to the Weight Norm Theory](https://whenwen.github.io/wd_blog/public/weight-decay-part-2.html)。

这说明，只看 weight norm 可能会漏掉非常重要的一部分动力学。Weight decay 确实控制了权重半径，但 gradient norm 本身仍然可能发生复杂变化。也就是说，weight decay 并不是简单地“让训练更平滑”，它可能改变甚至扰动梯度动力学结构。

### Effective Learning Rate 主要看的是 weight norm

论文中先引入 effective learning rate。对于 scale-invariant weight，有：

{{< math >}}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\boldsymbol{w}_t}
=
\frac{1}{\lVert \boldsymbol{w}_t \rVert_2}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\tilde{\boldsymbol{w}}_t}
{{< /math >}}

其中：

{{< math >}}
\tilde{\boldsymbol{w}}_t
=
\frac{\boldsymbol{w}_t}{\lVert \boldsymbol{w}_t \rVert_2}
{{< /math >}}

这说明 raw gradient norm 会受到 weight norm 的影响。权重越大，梯度越小；权重越小，梯度越大。这里要说清楚，**不是 raw grad norm 天然尺度无关**。恰恰相反，SMD 论文里的 Lemma 1 说的就是 raw gradient 会随着 weight scale 变化。

如果只看单位球面上的 SGD 更新，可以得到：

{{< math >}}
\tilde{\boldsymbol{w}}_{t+1}
=
\tilde{\boldsymbol{w}}_t
-
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2^2}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\tilde{\boldsymbol{w}}_t}
{{< /math >}}

于是定义：

{{< math >}}
\eta_{\mathrm{eff}}
=
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2^2}
{{< /math >}}

所以：

{{< math >}}
\tilde{\boldsymbol{w}}_{t+1}
=
\tilde{\boldsymbol{w}}_t
-
\eta_{\mathrm{eff}}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\tilde{\boldsymbol{w}}_t}
{{< /math >}}

这个式子非常重要。它告诉我们：在带 normalization 的网络中，原始学习率 $\eta$ 并不能直接表示模型在有效方向空间里的步长。因为同样的 $\eta$，如果 $\lVert \boldsymbol{w}_t \rVert_2$ 很大，那么单位球面上的有效学习率就会变小；如果 $\lVert \boldsymbol{w}_t \rVert_2$ 很小，那么有效学习率就会变大。

但是 effective learning rate 也有一个问题：

{{< math >}}
\eta_{\mathrm{eff}}
=
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2^2}
{{< /math >}}

它主要关注的是 weight norm。

也就是说，它回答的是：

> 当前权重半径会如何缩放学习率？

但它没有直接回答：

> 这一轮模型方向到底转了多少？

因为真正的方向变化不仅取决于 $\eta_{\mathrm{eff}}$，还取决于单位球面上的 gradient norm。

### Angular Update 把 grad norm 纳入了度量

因此，论文进一步提出了 angular update：

{{< math >}}
\Delta_t
=
\angle(\boldsymbol{w}_t,\boldsymbol{w}_{t+1})
=
\arccos
\left(
\frac{
\left\langle \boldsymbol{w}_t,\boldsymbol{w}_{t+1} \right\rangle
}{
\lVert \boldsymbol{w}_t \rVert_2
\lVert \boldsymbol{w}_{t+1} \rVert_2
}
\right)
{{< /math >}}

当学习率足够小时，有近似：

{{< math >}}
\Delta_t
\approx
\tan(\Delta_t)
=
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2}
\left\lVert
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\boldsymbol{w}_t}
\right\rVert_2
{{< /math >}}

再代入 scale-invariant gradient 的尺度关系：

{{< math >}}
\left\lVert
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\boldsymbol{w}_t}
\right\rVert_2
=
\frac{1}{\lVert \boldsymbol{w}_t \rVert_2}
\left\lVert
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\tilde{\boldsymbol{w}}_t}
\right\rVert_2
{{< /math >}}

可得：

{{< math >}}
\Delta_t
\approx
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2^2}
\left\lVert
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\tilde{\boldsymbol{w}}_t}
\right\rVert_2
{{< /math >}}

也就是：

{{< math >}}
\Delta_t
\approx
\eta_{\mathrm{eff}}
\cdot
\left\lVert
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\tilde{\boldsymbol{w}}_t}
\right\rVert_2
{{< /math >}}

这个式子比 effective learning rate 更完整。

因为 effective learning rate 只描述了：

{{< math >}}
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2^2}
{{< /math >}}

而 angular update 描述的是：

{{< math >}}
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2^2}
\times
\text{unit gradient norm}
{{< /math >}}

前者只是“系数”，后者才更接近“实际转动幅度”。

### 为什么 grad norm 比 weight norm 更有信息量？

这里需要说得精确一点。

不是说 raw gradient norm 天然比 weight norm 更“尺度无关”。事实上，论文中的 Lemma 1 明确告诉我们：

{{< math >}}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\alpha \boldsymbol{w}_t}
=
\frac{1}{\alpha}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\boldsymbol{w}_t}
{{< /math >}}

也就是说，raw gradient norm 本身也是会随 weight scale 改变的。

但是相比只看 weight norm，gradient norm 至少包含了 loss landscape 对当前方向的响应信息。weight norm 只告诉我们：

{{< math >}}
\text{当前半径有多大}
{{< /math >}}

而 gradient norm 告诉我们：

{{< math >}}
\text{在当前方向上，loss 还在多强地推动模型变化}
{{< /math >}}

尤其是 unit gradient norm：

{{< math >}}
\left\lVert
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\tilde{\boldsymbol{w}}_t}
\right\rVert_2
{{< /math >}}

它消除了 weight norm 的尺度影响，更接近单位球面上的真实优化信号。

因此，更准确的 insight 应该是：

> 在带 normalization 的网络中，weight norm 本身不直接代表模型功能；effective learning rate 只用 weight norm 去修正学习率，因此仍然不够。Angular update 进一步把 unit gradient norm 纳入进来，因此比单纯的 effective learning rate 更接近模型一次更新的真实有效改变量。

### Weight Decay 稳定的是半径，不一定稳定梯度动力学

从 SMD 角度看，SGD + weight decay 的更新可以写成：

{{< math >}}
\boldsymbol{w}_{t+1}
=
\boldsymbol{w}_t
-
\eta
\left(
\frac{\partial L}{\partial \boldsymbol{w}}
+
\lambda \boldsymbol{w}_t
\right)
{{< /math >}}

其中：

{{< math >}}
-
\eta
\frac{\partial L}{\partial \boldsymbol{w}}
{{< /math >}}

是切向更新，主要改变方向，并且会二阶地增大权重范数；

{{< math >}}
-
\eta\lambda\boldsymbol{w}_t
{{< /math >}}

是径向更新，主要缩小权重范数。

由于：

{{< math >}}
\left\langle
\boldsymbol{w}_t,
\frac{\partial L}{\partial \boldsymbol{w}}
\right\rangle
=
0
{{< /math >}}

所以不加 weight decay 时：

{{< math >}}
\lVert \boldsymbol{w}_{t+1} \rVert_2^2
=
\lVert \boldsymbol{w}_t \rVert_2^2
+
\left(
\eta
\left\lVert
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\boldsymbol{w}_t}
\right\rVert_2
\right)^2
>
\lVert \boldsymbol{w}_t \rVert_2^2
{{< /math >}}

这说明任务梯度天然会让 weight norm 增大。加入 weight decay 后，weight decay 提供径向收缩，可以阻止 weight norm 无限增大，使其进入一个平衡半径。

但是 weight norm 进入平衡，并不意味着 gradient norm 也平稳。因为 gradient norm 不只是由 weight norm 决定，还受到当前方向、loss landscape、batch noise、优化器状态等因素影响。

所以，如果我们只观察：

{{< math >}}
\lVert \boldsymbol{w}_t \rVert_2
{{< /math >}}

可能会看到一个很漂亮的平衡现象；

但如果进一步观察：

{{< math >}}
\left\lVert
\frac{\partial L}{\partial \boldsymbol{w}}
\right\rVert_2
{{< /math >}}

可能会发现梯度动力学仍然在抖动、上升或者发生结构性变化。

这就是为什么 angular update 比 effective learning rate 更有解释力：

{{< math >}}
\eta_{\mathrm{eff}}
=
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2^2}
{{< /math >}}

只关心半径；而：

{{< math >}}
\Delta_t
\approx
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2}
\left\lVert
\frac{\partial L}{\partial \boldsymbol{w}}
\right\rVert_2
{{< /math >}}

同时关心半径和梯度。

### 这个 insight 的落点

这一部分可以得到一个很重要的 insight：

> 对于带 normalization 的网络，只看 weight norm 或 effective learning rate 仍然不够。Weight norm 主要描述的是半径动力学，而模型真正的有效变化发生在方向空间。Angular update 把 weight norm 和 gradient norm 同时纳入度量，因此更接近一次梯度更新对模型函数造成的真实改变量。

更进一步说：

> Weight decay 可以稳定 weight norm，但不一定稳定 grad norm。它解决的是半径无限增长的问题，却可能引入新的梯度动力学变化。因此，SMD 不应该只看半径平衡，还应该进一步观察 gradient norm，尤其是 unit gradient norm，以及最终的 angular update。

用一句话概括：

{{< math >}}
\boxed{
\text{Effective learning rate 主要看 weight norm，Angular update 同时看 weight norm 和 grad norm。}
}
{{< /math >}}

## Insight 2：权重范数收敛不等于优化收敛

SMD 这篇论文里还有一个很关键的洞察：

> Weight norm convergence is not equivalent to weight convergence.

也就是说，权重范数稳定了，不代表权重本身不动了；权重本身不动了，也不一定代表 loss 已经收敛到了一个好解。

在带 normalization 的网络里，我们至少要区分三件事：

{{< math >}}
\lVert \boldsymbol{w}_t \rVert_2
\text{ 收敛}
{{< /math >}}

{{< math >}}
\boldsymbol{w}_t
\text{ 收敛}
{{< /math >}}

{{< math >}}
L(\boldsymbol{w}_t)
\text{ 收敛到一个好解}
{{< /math >}}

这三件事不是一回事。SMD 最有价值的地方之一，就是把这几个容易混在一起的“收敛”拆开了。

### 没有 Weight Decay 时，权重范数会不断增大

对于 scale-invariant weight，有：

{{< math >}}
L(\alpha\boldsymbol{w})
=
L(\boldsymbol{w})
{{< /math >}}

于是任务梯度满足：

{{< math >}}
\left\langle
\boldsymbol{w}_t,
\frac{\partial L}{\partial \boldsymbol{w}}
\right\rangle
=
0
{{< /math >}}

也就是说，任务梯度和权重方向垂直。

如果使用普通 SGD，且没有 weight decay：

{{< math >}}
\boldsymbol{w}_{t+1}
=
\boldsymbol{w}_t
-
\eta
\frac{\partial L}{\partial \boldsymbol{w}}
{{< /math >}}

那么：

{{< math >}}
\lVert \boldsymbol{w}_{t+1} \rVert_2^2
=
\lVert \boldsymbol{w}_t \rVert_2^2
+
\left(
\eta
\left\lVert
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\boldsymbol{w}_t}
\right\rVert_2
\right)^2
>
\lVert \boldsymbol{w}_t \rVert_2^2
{{< /math >}}

这说明，对于 normalized weight 来说，任务梯度虽然是切向更新，但它会二阶地增加权重范数。论文第三部分也明确指出，由于梯度与权重正交，不带 weight decay 时，weight norm 会持续增加。

用 SMD 的话说就是：

{{< math >}}
\text{SGD 梯度负责转方向，但它会顺手把半径撑大。}
{{< /math >}}

### Weight norm 增大，会让 raw gradient 变小

论文 Lemma 1 还有另一个重要结论：

{{< math >}}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\alpha \boldsymbol{w}_t}
=
\frac{1}{\alpha}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\boldsymbol{w}_t}
{{< /math >}}

因此，如果定义：

{{< math >}}
\tilde{\boldsymbol{w}}_t
=
\frac{\boldsymbol{w}_t}{\lVert \boldsymbol{w}_t \rVert_2}
{{< /math >}}

那么：

{{< math >}}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\boldsymbol{w}_t}
=
\frac{1}{\lVert \boldsymbol{w}_t \rVert_2}
\left.
\frac{\partial L}{\partial \boldsymbol{w}}
\right|_{\boldsymbol{w}=\tilde{\boldsymbol{w}}_t}
{{< /math >}}

也就是：

{{< math >}}
\lVert \boldsymbol{g}_t \rVert_2
=
\frac{1}{\lVert \boldsymbol{w}_t \rVert_2}
\lVert \tilde{\boldsymbol{g}}_t \rVert_2
{{< /math >}}

这里 $\boldsymbol{g}_t$ 是 raw gradient，$\tilde{\boldsymbol{g}}_t$ 是 unit gradient。

所以，raw gradient norm 变小，不一定意味着模型真的接近最优点。它也可能只是因为：

{{< math >}}
\lVert \boldsymbol{w}_t \rVert_2
\uparrow
{{< /math >}}

导致：

{{< math >}}
\lVert \boldsymbol{g}_t \rVert_2
\downarrow
{{< /math >}}

论文也正是基于这一点指出，如果 unit gradient norm 没有变，那么 increasing weight norm 本身就会导致 smaller gradient norm。

这就给了我们一个非常重要的诊断视角：

> 如果只看 raw gradient norm，可能会误以为训练快到驻点了；但实际上只是 weight norm 变大，把 raw gradient 压小了。

### Effective Learning Rate 下降会造成“驻点假象”

归一化网络里，单位球面上的 effective learning rate 是：

{{< math >}}
\eta_{\mathrm{eff}}
=
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2^2}
{{< /math >}}

也就是说，权重范数越大，模型在单位球面上的实际步长越小。

不带 weight decay 时，前面已经看到 $\lVert \boldsymbol{w}_t \rVert_2$ 会不断增大。于是：

{{< math >}}
\eta_{\mathrm{eff}}
=
\frac{\eta}{\lVert \boldsymbol{w}_t \rVert_2^2}
{{< /math >}}

会不断减小。最终会出现一种现象：

{{< math >}}
\Delta_t
\approx
\eta_{\mathrm{eff}}
\lVert \tilde{\boldsymbol{g}}_t \rVert_2
{{< /math >}}

越来越小。

这个时候模型看起来像是“走不动了”。在 loss 曲线上，这可能表现为 loss 下降变慢，甚至长时间平台。如果只看 raw gradient norm，它也可能越来越小，于是我们很容易误判：

> 模型已经接近一个驻点了。

但 SMD 视角会提醒我们：

> 这不一定是真正的 loss landscape 驻点，而可能是 weight norm 增大导致 effective learning rate 衰减后的驻点假象。

论文第三部分引用相关工作的观点指出，没有 weight decay 但带 BN 的 GD/SGD，存在一种风险：模型可能不是通过真正降低 loss 到达驻点，而是通过增加 weight norm、降低 effective learning rate，让优化过程看起来停下来。

不过这里要说准确一点。论文更精确的意思不是“GD 一定不会、SGD 一定会”，而是 full GD 在某些条件下可以避免这个问题，并在单位球面上收敛到驻点；但 SGD 情况仍然需要更复杂的学习率衰减设计。

所以这里的重点是：

{{< math >}}
\text{归一化网络里，驻点判断必须放到单位球面上看。}
{{< /math >}}

### Weight norm 收敛不等于 weight 收敛

传统直觉里，我们可能会以为：

{{< math >}}
\lVert \boldsymbol{w}_t \rVert_2
\text{ 稳定}
\Rightarrow
\boldsymbol{w}_t
\text{ 快稳定了}
{{< /math >}}

但 SMD 论文明确指出，这个推理是不对的。

权重可以写成：

{{< math >}}
\boldsymbol{w}_t
=
r_t\boldsymbol{u}_t
{{< /math >}}

其中：

{{< math >}}
r_t
=
\lVert \boldsymbol{w}_t \rVert_2
{{< /math >}}

是半径；

{{< math >}}
\boldsymbol{u}_t
=
\frac{\boldsymbol{w}_t}{\lVert \boldsymbol{w}_t \rVert_2}
{{< /math >}}

是方向。

对于 normalized weight，模型函数主要由方向 $\boldsymbol{u}_t$ 决定，而不是由半径 $r_t$ 决定。因此可能出现：

{{< math >}}
r_t
\rightarrow
r^*
{{< /math >}}

但 $\boldsymbol{u}_t$ 仍然在单位球面上持续运动。

也就是说：

{{< math >}}
\lVert \boldsymbol{w}_t \rVert_2
\text{ 收敛}
{{< /math >}}

只代表半径稳定，不代表权重方向停止变化，更不代表模型函数停止变化。

论文在第三部分最后说得非常清楚：the convergence of weight norm is not equivalent to convergence of weight。这句话其实是全文最关键的洞察之一。

它把“优化收敛”拆成了两个不同问题：

{{< math >}}
\text{半径是否收敛？}
{{< /math >}}

{{< math >}}
\text{方向是否收敛？}
{{< /math >}}

SMD 研究的是前者如何在 normalization + weight decay + SGD 下形成平衡；而模型训练是否真正结束，还要看方向空间上的运动和 loss 是否继续下降。

### Unit gradient norm 稳定，也可以导致 weight norm 稳定

更进一步，论文指出：稳定的 unit gradient norm，也可以让 weight norm 收敛。

这是一个很重要的反直觉点。很多前人解释 equilibrium 时，会默认认为：

{{< math >}}
\boldsymbol{w}_t
\approx
\boldsymbol{w}_{t+1}
{{< /math >}}

也就是权重本身快不动了，所以权重范数稳定了。

但 SMD 论文说，不需要这样。只要单位球面上的梯度统计比较稳定，weight norm 就可以收敛到一个平衡半径。

也就是说，weight norm 的收敛可以来自：

{{< math >}}
\text{unit gradient norm 稳定}
{{< /math >}}

而不是来自：

{{< math >}}
\text{优化已经到达最优点}
{{< /math >}}

论文原文也说，steady unit gradient norm can also make weight norm converge，并且这并不依赖优化已经到达 optimum solution。

这就解决了一个很大的矛盾：如果 equilibrium 是因为优化已经结束，那么更新幅度应该趋近于 0。但前人推导又说，在 equilibrium 下 angular update 仍然是一个由超参数决定的非零值，比如：

{{< math >}}
\Delta^*
\approx
\sqrt{2\eta\lambda}
{{< /math >}}

这说明模型仍然在单位球面上运动。

所以更合理的解释是：

{{< math >}}
\text{equilibrium 不是优化停止，而是半径动力学达到平衡。}
{{< /math >}}

### 这个 insight 对训练诊断有什么意义？

这个 insight 可以直接转化为一个训练诊断原则：

> 不要只用 raw gradient norm 或 weight norm 判断模型是否真的收敛。

如果看到：

{{< math >}}
\lVert \boldsymbol{g}_t \rVert_2
\downarrow
{{< /math >}}

要进一步问：

{{< math >}}
\lVert \tilde{\boldsymbol{g}}_t \rVert_2
\text{ 是否也下降？}
{{< /math >}}

因为：

{{< math >}}
\lVert \boldsymbol{g}_t \rVert_2
=
\frac{
\lVert \tilde{\boldsymbol{g}}_t \rVert_2
}{
\lVert \boldsymbol{w}_t \rVert_2
}
{{< /math >}}

如果 raw gradient 下降只是因为 $\lVert \boldsymbol{w}_t \rVert_2$ 变大，那么这不一定意味着模型接近真正的驻点。

同样，如果看到 $\lVert \boldsymbol{w}_t \rVert_2$ 稳定，也要进一步问：

{{< math >}}
\Delta_t
=
\angle(\boldsymbol{w}_t,\boldsymbol{w}_{t+1})
{{< /math >}}

是否仍然非零。

如果 angular update 仍然稳定非零，那么模型仍然在方向空间里运动，只是半径不变了。

### 这个 insight 的落点

这一部分的核心 insight 是：

> 在归一化网络中，raw gradient norm 变小可能只是 weight norm 增大造成的；effective learning rate 降低会让模型看起来像到达驻点，但这可能只是驻点假象。SMD 的重要贡献在于指出：weight norm 收敛不等于 weight 收敛，更不等于优化收敛。只要 unit gradient norm 在局部统计上稳定，weight norm 就可以达到平衡，而模型方向仍然可以继续更新。

用一句话总结：

{{< math >}}
\boxed{
\text{半径收敛不是优化收敛，raw gradient 变小也不一定是真驻点。}
}
{{< /math >}}

或者更口语一点：

> 在 normalized network 里，模型可能不是“学到头了”，而只是“半径变大导致步子变小了”。SMD 的价值就在于，它把这种驻点假象和真正的优化收敛区分开了。

### insight2的原文

第二个 insight 论文里也说的比较明确，如果使用 GD，就不会因为随机噪声衰减到驻点，而使用 SGD 就会导致到驻点。因为随着训练进程的进行，有效学习率在降低，最后可能到一个看起来梯度 0 的点。体现在 loss 曲线上就是 loss 根本下不去，看似已经最低了。论文第三部分最后说的很清晰，论文实际上发现了权重范数收敛并不等价于权重收敛。而且论文还说了梯度范数收敛也可以让权重范数收敛，而且不会让梯度 = 0 这种驻点假象出现。

### insight1的原文
我因为懒得敲公式，直接把insight丢给gpt，让他帮我完整的搞一篇blog，所以这里贴一下原文，也方便去check思想。

我现在要回到论文本身去思考一些insight，首先，思考归一化与权重衰减的联合作用这个地方，这里有一句很关键的话叫做直，梯度分量 ∂L/∂w 总是倾向于增大权重范数，而权重
衰减提供的梯度分量总是倾向于减小权重范数。这里要注意他的描述，权重衰减能对冲掉梯度更新时 权  重  的范数，因此不会让权重的范数无限增大。但对于梯度的范数本身而言又如何呢？如果你进入https://whenwen.github.io/wd_blog/public/weight-decay-part-2.html就会看到他的一张图，加weight-decay，虽然让weightnorm逐渐减小最后趋于稳定，但是会导致grad norm抖动上升。这说明wd破坏了动力学结构。然后可以看我发的第一张图里面的蓝色方框对比，用有效学习率度量的时候，主要关注weight的二范数，而用本文定义角变化量的时候，主要关注的是grad的二范数。那我们基于刚才说的这两点（图和公式）就能double check出来一个insight：grad的二范数比weight的二范数要“有价值”。这个insight是有些符合直觉的，因为参数本身是尺度有关的，会随着尺度膨胀，就导致他有效学习率的度量也没那么有效。但是梯度是尺度无关的，所以度量的更合理。

## 参考文献

1. Ruosi Wan, Zhanxing Zhu, Xiangyu Zhang, Jian Sun. [Spherical Motion Dynamics: Learning Dynamics of Neural Network with Normalization, Weight Decay, and SGD](https://arxiv.org/abs/2006.08419). arXiv:2006.08419, 2020.
2. Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, Percy Liang. [Fantastic Pretraining Optimizers and Where to Find Them 2.2: The Hitchhiker's Guide to the Weight Norm Theory](https://whenwen.github.io/wd_blog/public/weight-decay-part-2.html).
