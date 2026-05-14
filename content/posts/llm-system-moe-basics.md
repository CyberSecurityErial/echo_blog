---
date: '2026-05-14T01:16:50+08:00'
draft: true
title: 'MoE 基础知识'
categories: ["AI"]
tags: ["LLM", "LLM System", "MoE"]
series: ["LLM System"]
math: true
---

这篇文章就是忘了回来查一查用的。
有时候dirty work做多了反而会忘记一开始的系统是如何构建出来的。

原来的transformer是attn+FFN,FFN其实就是MLP。moe就是把这里的FFN换了一下，变成了attn+router+choose(FFNs)+avg。
也就是非常简单只激活一部分FFN参数，最后把输出平均一下。有点像联邦学习里面选择topk进行avg一样（不要问我为什么会想到联邦学习）

计算量。
moe是逐个输入token的，什么意思呢，我们知道保存一次推理的token形状为[B,L,Hd],B代表batch，L代表位置，Hd代表embeddim。而计算moe的时候输入的是固定了b和l的。
也就是一个req里面一个位置上的token，形状就是[Hd]。

先看一般的dense，就是把x先升维到I。
u = xW_up (W_up Hd×I)
g = xW_gate (W_gate Hd×I)
z = silu(g)⊙u
y = zW_down (W_down I×Hd)
## 计算量
这里I是FFN的中间Size，计算量涉及到三个矩阵乘，一次乘加算2FLOPS的话一个token计算量6HI。N个token就是6NHI

moe觉得直接升维到I太奇怪了，太浪费。就把一个大FFN拆成了多个小FFN，一次只让一部分FFN激活。一个FFN叫一个Expert，数量一般写E。参数量的话没道理比FFN更小，如果每个Expert都是一个原FFN的形状，那么参数量直接×E。计算量却不用，因为不是所有的Expert都计算。计算Expert就是计算FFN，此处不赘述。

路由过程也是对输入做矩阵乘，全是矩阵乘，没有别的操作。
router_guide = xW_router [Hd × E]
router_guide最后形状就是E，E就是给每个专家打分，给每个token取一个自己的topk代表你选这几个Expert，最后就是这几个Expert来激活并给你综合结果。

这过程问题就是你不一定激活哪个Expert，因为激活谁就让谁算，所以读取Expert的时候就变成了一个完全随机的内存访问问题，扩展到多卡就更是完全随机的通信问题。
## 通信量
分布式ep、moe的计算可以划分成几个阶段。本地计算route，dispatch，compute，combine。分别计算路由对象，做一次a2a分发出去（因为a2a通信kernel感知不到router_guide的信息。
发出去之后如果dst收到了就做FFN，最后combine把数据发到原来GPU做一次avg。

最普通的实现就是有一个token，router算完这个token的k个expert就往外发。但是这样还是效率太低了。因为一次只能处理一个。我们之前说过token的经典形状是[B,T,Hd]分别对应batch，sequence和hiddendim。经过attention算完之后token的形状还是[B,T,Hd]，但是这个时候因为token已经带位置信息了，所以可以并行的去计算，不会丢信息。有了这里的铺垫，那么为了让实际工程中token & router的计算密度更高，可以把B和T两个维度做展平，然后一次喂多个token给router，router会给每个token都计算出他们的topk expert，然后把激活了expert i的所有token都打包发到expert，这样做在expert激活较为集中 or expert比较少的时候能显著降低通信量。见下面的(文字？)图解：

低效率：
token0 -> expert0 expert1 -> sendE0T0 sendE1T0
token1 -> expert1 expert2 -> sendE1T1 sendE2T1
token2 -> expert2 expert0 -> sendE2T2 sendE0T2
高效率
token(0, 1, 2) -> expert( (0,1), (1,2) , (2,0) ) -> sendE0T0T2 sendE1T0T1 sendE2T1T2

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

一tokenMOE通信量计算其实就是E×Hd×k×2
