---
date: '2026-07-05T00:00:00+08:00'
draft: false
title: 'LLM System: 通信计算融合算子实现 01 - Ampere 风格的 GEMM + ReduceScatter'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "CUDA", "GEMM", "ReduceScatter", "Communication Overlap", "Distributed Training"]
series: ["LLM System", "通信计算融合算子实现"]
series_order: 1
weight: 1
math: true
---

## flux实现：sm80 gemm+rs
看flux的论文来说，原理并不难，修改的是ffn的第二个gemm的epi阶段，让epi阶段的写回操作变成写入reducescatter的通信buffer，相当于做了一次零拷贝。
几个关键的点：1. 如何用cutlass自定义epi阶段。2. 用了什么ptx。3. 清零？
## cutlass自定义epilogue--sm80
常规的epi，写回gmem，这里改成写在scatter-aware-memory（我自己起的名字）。cutlass封装自定义epi的逻辑抽象是：输出tile存到哪里，以及做什么运算，这分别是两个模版类：
using EVT_D = decltype(this->evt_d(kparams));
using StoreD = decltype(this->evt_store_d(kparams));
using EVT = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_D>;
StoreD就是规定了怎么存，EVTD就是规定了怎么算。
EVT全称epilogue visitor tree就是定义了一个epi阶段的计算-访存操作流，用树状图的形式保存epi阶段的一堆操作（做成图应该是为了方便接入nvcc？不懂为啥非得在这强调是tree）一个树节点（操作）是一个visitor。
### custom_evt_d()
EVT_Compute0 = alpha * accumulator
代码里对应 VisitorCompute<cutlass::multiplies, ...>，两个输入是 VisitorScalarBroadcast<ElementAccumulator> 和 VisitorAccFetch。第一个负责广播 alpha，第二个v从 GEMM accumulator 里拿当前 tile 的结果，然后 multiply

EVT_Compute1 = beta * C + EVT_Compute0 = beta * C + alpha * accumulator
代码里对应 VisitorAuxLoadGemmk 先把 C/bias 读进 epilogue，然后 VisitorCompute<cutlass::multiply_add, ...> 做 beta * C + alpha * acc。因为CUTLASS 2.x 没有 SM90 那种 SrcFetch 替代物，所以这里用 VisitorAuxLoadGemmk 来读 C。
### evt_store_d()
会自动构建StoreD = VisitorAuxStoreScatter<...>，拿到当前数据的metadata（多卡环境下这个就是某rank的某阶段的某一块数据，以及来自哪里要stor到哪里）
这里还会设计一个barrierflag，保证rs操作的时候rs-worker能拿到对的tile。（这个barrierflag怎么实现的？）
### gemm template对自定义epi的声明
cutlass::gemm::kernel::GemmkWithVisitor<..., decltype(params.evt()), ...>
相当于告诉gemm模版这里有个visitor，需要经过一遍。
### swizzle
swizzle决定了cta-tile的调度顺序。也就是怎么把blocktile指派给不同rank。这个是防止所有rank都往一个rank的某个同一chunk写引发性能问题。
swizzle的编排，如何实现？
https://chatgpt.com/g/g-p-6a4a755b6da48191b4bf6cda6bca24fd-flux-tong-xin-ji-suan-rong-he-suan-zi-yan-xi/c/6a450bd7-a0ac-83ee-bb97-0c0dfcdbce25
### ptx应用位置
codex resume 019f32ab-30e4-7cf3-b9fb-e0d7389d24d0

## 自己的类flux实现
### test case
跑了一下发现自己的flux测出来效果远低于flux原生，甚至低于cutlass+nccl的baseline。跑了十组case发现自己测出来的时间非常不稳，而且和size关系不大。因此看起来是流同步一类的开销。另外还有个开销是要保证reduce的buffer一开始是清零状态，非0的话reduce操作会出错。

所以设计了这样一个消融方案去测：
因为一次iter可以这么拆：
  A = cudaMemsetAsync(output)
  B = cudaStreamSynchronize after clear
  C = host thread barrier after clear
  D = launch fused GEMMRS kernel
  E = cudaStreamSynchronize after kernel
  F = host thread barrier after kernel

其中ABC就是我说的清零clear操作。
然后消融测试：
  kernel-batched      = D D D D ... 
  kernel-sync         = (D + E + F) ...
  clear-only          = (A + B + C) ...
  clear-kernel-sync   = (A + B + C + D + E + F) ...

然后就很好测每个阶段都多少时间了。
最后测出来发现主要还是流同步那边花时间多，clear操作是几十us，占整个算子的不到10%吧。
但是不代表就没事了，如果清零之后不做同步，gemmrs算子就不能开始。如果做了同步就会很慢。这里要做的除了stream级别的clear-gemmrs保序，还有cross rank的同步。不可能一个rank已经开始gemm了，另一个rank还在清零。所以必须得等所有rank做完以后host这边做一个收口，确定都zero了再去gemmrs。这种比较普通的思路，开销不可接受，所以就研究了一下flux怎么实现的这部分。

codex resume 019f2056-6ae2-7591-8984-7bd880a6ef84