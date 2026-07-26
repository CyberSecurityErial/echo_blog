---
date: '2026-07-26T18:43:54+08:00'
draft: false
title: 'LLM Theory: MuP 03 - 为什么MoE模型训练起来有难度？MuP视角'
categories: ["LLM Theory"]
tags: ["LLM", "LLM Theory", "Pretraining", "MuP", "MoE"]
series: ["LLM Theory", "MuP"]
series_order: 3
weight: 3
math: true
---

众所周知MoE有一个Router，也就是门控单元，基于topk选择路由到的专家。训练稳定的基本思想是希望随着宽度d增大，模型的activation、grad、delta loss、delta weight变化都比较小。

{{< math >}}
\operatorname{RMS}(h_l)=\Theta(1),\quad
\operatorname{RMS}(\Delta h_l)=\Theta(1),\quad
\Delta L=\Theta(1).
{{< /math >}}

不好分析的根源是因为topk不光滑，所以找不到一个有界的全局利普希茨常数。

也就是假设两个输入的x很接近，但是在router这里发生了跳变，导致L也就是$\lVert f(x_2)-f(x_1)\rVert/(x_2-x_1)$趋近于无穷大。

所以说topk没有全局的LipschitzBound。那么为什么lipschitz常数对mup会有影响，原因是对于某个token的一次backward里面参数更新，如果weight更新后导致路由切换，delta y的计算就不只是简单的$\Delta y\approx J_\theta y\Delta\theta$，而是两个expert相减。$\Delta y=F_j(h)-F_i(h)$. 这样可能会导致，$\lVert\Delta x\rVert\to0,\lVert\Delta y\rVert=\Theta(1)$.这个导致的最直接问题就是我们没有办法再去用一阶泰勒近似delta y了，也就是用dense的方式，没有办法用局部梯度去正确描述某个token的路由表随参数更新导致变化的这个事件，也就不能囊括所有的梯度更新类型。

从另一个角度，如果强行进行泰勒展开（当然肯定有各种不严谨，我是工程民科，凑合看看算了），就有$y(\theta+\Delta\theta)-y(\theta)=J_\theta y\Delta\theta+o(\lVert\Delta\theta\rVert)$. 但最大的问题是$\lVert y(\theta+\Delta\theta)-y(\theta)\rVert=\Theta(1)$,$\frac{\lVert y(\theta+\Delta\theta)-y(\theta)\rVert}{\lVert\Delta\theta\rVert}$所以分子分母不同阶。taylor余项也不是无穷小的。

要是想做MoE的mup，就得想办法去控制它由于router切换导致的跳变大小，也就是说必须要建模触发router跳变的条件，以及给跳变以后专家输出的diff加上一个bound。路由跳变对均方特征的贡献可以如下描述：

{{< math >}}
\mathbb{E}\left[\lVert\Delta y_{\mathrm{switch}}\rVert^2\right]
\lesssim
\int_0^{2\varepsilon}q(u)f_\gamma(u)\,\mathrm{d}u.
{{< /math >}}

f描述有多少token距离路由边界为u，q表征输出跳变的距离，2sigma代表参数更新能跨越多宽的边界。那么这件事容易吗？

我们之前只是假设了输入不变，在mup前面的系列里面，mup的目标之一是要让中间的特征发生一些非退化的显著变化，$\Delta h=\Theta(1)$。所以即使router参数冻结了，h的变化也会导致路由切换。这个问题会让特征更新的尺度推导变得困难。

总而言之，传统 Dense μP 能闭环，是因为给定每层的：

{{< math >}}
\operatorname{RMS}(h),\quad
\operatorname{RMS}(\delta),\quad
\operatorname{RMS}(\Delta W),
{{< /math >}}

就能递推出下一步的特征更新：

{{< math >}}
\Delta h\approx J_W h\Delta W,
{{< /math >}}

而这个更新尺度仍只由矩阵维度、初始化尺度和学习率决定。

MoE 不闭环，是因为即使这些 RMS 都已知，仍然无法确定输出更新：

{{< math >}}
\Delta y
=
\Delta y_{\mathrm{smooth}}
+
\Delta y_{\mathrm{switch}}.
{{< /math >}}

第二项还取决于两个额外变量：

{{< math >}}
P(\text{route switch})
{{< /math >}}

和

{{< math >}}
\lVert F_i-F_j\rVert.
{{< /math >}}

前者由路由 margin 分布决定，后者由专家分化程度决定；它们都不能从参数 RMS、梯度 RMS 和宽度直接推出。因此原来的 μP 递推方程少了状态变量，无法仅靠已有尺度计算下一步尺度

margin和专家分化程度又是只能测出来，但是不能像传统mup一样去预测他们随着模型尺度增长所产生的变化，所以这个东西理论上存在空洞。而且理论上margin和专家分化应该是一个联合分布。必须同时去估计易跳变的token和跳变后的范围。margin 决定专家看哪些 token；

token 分配决定专家如何分化；

专家分化又决定 Router 梯度；

Router 梯度再改变 margin。

即使当前时刻测得：

{{< math >}}
P_t(\gamma,D),
{{< /math >}}

也不一定能仅凭它和参数 RMS 推出：

{{< math >}}
P_{t+1}(\gamma,D).
{{< /math >}}

还可能需要加入：

{{< math >}}
P(h\mid e),\quad
\operatorname{Cov}(F_e,F_j),\quad
\operatorname{Cov}(F_e,\delta),\quad
\text{优化器状态},
{{< /math >}}

于是状态变量可能继续增加。

有的论文用了一些很高级的数学工具，https://arxiv.org/abs/2605.14200 

不过我也不知道是否work，，，

## 共享专家

从mup角度共享专家能否提高稳定性？留作思考问题。
