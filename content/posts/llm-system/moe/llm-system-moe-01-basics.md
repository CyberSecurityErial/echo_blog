---
date: '2026-05-14T01:16:50+08:00'
draft: true
title: 'LLM System: MoE 01 - 基础知识'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "MoE"]
series: ["LLM System", "MoE"]
series_order: 1
weight: 1
math: true
aliases:
  - /posts/llm-system-moe-basics/
---

这篇文章就是忘了回来查一查用的。
有时候dirty work做多了反而会忘记一开始的系统是如何构建出来的。

原来的transformer是attn+FFN,FFN其实就是MLP。moe就是把这里的FFN换了一下，变成了attn+router+choose(FFNs)+avg。
也就是非常简单只激活一部分FFN参数，最后把输出平均一下。有点像联邦学习里面选择topk进行weight avg一样（不要问我为什么会想到联邦学习）
## 前向计算量
计算量。
moe是逐个输入token的，什么意思呢，我们知道保存一次推理的token形状为[B,L,Hd],B代表batch，L代表位置，Hd代表embeddim。而计算moe的时候输入的是固定了b和l的。
也就是一个req里面一个位置上的token，x形状就是[1×Hd]。

先看一般的dense，swiglu为例，就是把x先升维到I。
u = xW_up (W_up Hd×I)
g = xW_gate (W_gate Hd×I)
z = silu(g)⊙u
y = zW_down (W_down I×Hd)

这里I是FFN的中间Size，计算量涉及到三个矩阵乘，一次乘加算2FLOPS的话一个token计算量6HI。N个token就是6NHI

moe觉得直接升维到I太奇怪了，太浪费。就把一个大FFN拆成了多个小FFN，一次只让一部分FFN激活。一个FFN叫一个Expert，数量一般写E。参数量的话没道理比FFN更小，如果每个Expert都是一个原FFN的形状，那么参数量直接×E。计算量却不用，因为不是所有的Expert都计算。计算Expert就是计算FFN，此处不赘述。

路由过程也是对输入做矩阵乘，全是矩阵乘，没有别的操作。
router_guide = xW_router [Hd × E]
router_guide最后形状就是E，E就是给每个专家打分，给每个token取一个自己的topk代表你选这几个Expert，最后就是这几个Expert来激活并给你综合结果。

这过程问题就是你不一定激活哪个Expert，因为激活谁就让谁算，所以读取Expert的时候就变成了一个完全随机的内存访问问题，扩展到多卡就更是完全随机的通信问题。

per token过程，依然按照swiglu为例：
router_score = x W_router          # [1, E]
topk_ids, topk_weights = TopK(router_score)

for i in topk_ids:
    u_i = x W_up_i                 # [1, I]
    g_i = x W_gate_i               # [1, I]
    z_i = silu(g_i) ⊙ u_i          # [1, I]
    y_i = z_i W_down_i             # [1, H]

y = Σ_i topk_weights_i * y_i       # [1, H]
router会得到一组logits，会过一下softmax，topk选出里面最大的k个logits，然后用这个logits在y这里加权。或者给logits再做一次归一化，再加权。都差不多。

每个token对于每个expert来说计算量确实还是6HI，k个expert那就是6HIK，加上计算router的操作还有2HE的计算量，总计算量是6HIK+2HE/token。如果用其他的active会有其他形式。大差不差。

## 前向通信量
分布式ep、moe的计算可以划分成几个阶段。本地计算route，dispatch，compute，combine。分别计算路由对象，做一次a2a分发出去（因为a2a通信kernel感知不到router_guide的信息。
发出去之后如果dst收到了就做FFN，最后combine把数据发到原来GPU做一次weight avg。

最普通的实现就是有一个token，router算完这个token的k个expert就往外发。但是这样还是效率太低了。因为一次只能处理一个。我们之前说过token的经典形状是[B,T,Hd]分别对应batch，sequence和hiddendim。经过attention算完之后token的形状还是[B,T,Hd]，但是这个时候因为token已经带位置信息了，所以可以并行的去计算，不会丢信息。有了这里的铺垫，那么为了让实际工程中token & router的计算密度更高，可以把B和T两个维度做展平，然后一次喂多个token给router，router会给每个token都计算出他们的topk expert，然后把激活了expert i的所有token都打包发到expert，这样做在expert激活较为集中 or expert比较少的时候能显著降低建立跨机通信的次数，并且更容易打满带宽。见下面的(文字？)图解：

低效率：
token0 -> expert0 expert1 -> sendE0T0 sendE1T0
token1 -> expert1 expert2 -> sendE1T1 sendE2T1
token2 -> expert2 expert0 -> sendE2T2 sendE0T2
高效率
token(0, 1, 2) -> expert( (0,1), (1,2) , (2,0) ) -> sendE0T0T2 sendE1T0T1 sendE2T1T2

这样打包的另一个好处是可以用groupGEMM而不是小型的GEMV，这个要后面进行讨论。

附： attention过程中的shape，其实最后是没变的：
X
[B, T, Hd]

Q/K/V projection
[B, T, Hd]

split heads
[B, T, num_heads, head_dim]

transpose for attention
[B, num_heads, T, head_dim]

attention score
[B, num_heads, T, T]

attention output per head
[B, num_heads, T, head_dim]

merge heads
[B, T, Hd]

output projection
[B, T, Hd]

## Group Route？
这是我自己起的名，本来是想说GroupGEMM，但是感觉标题叫GroupGEMM只能说明计算上的事情，不能提到通信。所以就单纯把标题起成Group这个操作，计算和通信都提及。

前向通信的时候提到了将attn过的token在BT两维度展平，并按组给router计算路由，这样做会有计算&通信效率上的提升。这个说法很凭感觉，我们还是拿出实际数据或者profilng方案（脑测）check一下这件事。

先看比较好分析的通信。
假设打包一次给router处理S个。
在不打包的时候，逐个token的跨机建立通信的次数是K，通信的数据量是KH。那么对于S个token通信的数据量是KHS，建立跨机的次数是KS。
在打包的时候，建立跨机通信的次数不一定，这要取决于这S个token到底遍历了多少个expert。最坏的情况就是expert很多，S个token需要KS个expert。但是这也只是和不打包的方案相等，是一个很宽松的上界。通信的数据量不变，但是节省了大量建立通信所需的overhead时间。

再看略有复杂的计算。
