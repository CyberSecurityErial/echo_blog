---
date: '2026-07-05T00:00:00+08:00'
draft: false
title: 'LLM System: 通信计算融合算子实现 01 - sm80GEMM + ReduceScatter'
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

```cpp
using EVT_D = decltype(this->evt_d(kparams));
using StoreD = decltype(this->evt_store_d(kparams));
using EVT = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_D>;
```

StoreD就是规定了怎么存，EVTD就是规定了怎么算。
EVT全称epilogue visitor tree就是定义了一个epi阶段的计算-访存操作流，用树状图的形式保存epi阶段的一堆操作（做成图应该是为了方便接入nvcc？不懂为啥非得在这强调是tree）一个树节点（操作）是一个visitor。

### custom_evt_d()

```text
EVT_Compute0 = alpha * accumulator
```

代码里对应 VisitorCompute<cutlass::multiplies, ...>，两个输入是 VisitorScalarBroadcast<ElementAccumulator> 和 VisitorAccFetch。第一个负责广播 alpha，第二个v从 GEMM accumulator 里拿当前 tile 的结果，然后 multiply

```text
EVT_Compute1 = beta * C + EVT_Compute0 = beta * C + alpha * accumulator
```

代码里对应 VisitorAuxLoadGemmk 先把 C/bias 读进 epilogue，然后 VisitorCompute<cutlass::multiply_add, ...> 做 beta * C + alpha * acc。因为CUTLASS 2.x 没有 SM90 那种 SrcFetch 替代物，所以这里用 VisitorAuxLoadGemmk 来读 C。

### evt_store_d()

会自动构建StoreD = VisitorAuxStoreScatter<...>，拿到当前数据的metadata（多卡环境下这个就是某rank的某阶段的某一块数据，以及来自哪里要stor到哪里）
这里还会设计一个barrierflag，保证rs操作的时候rs-worker能拿到对的tile。做法就是给每个tile配一个全局唯一标识符，然后只需要描述tile的生产消费关系即可。

```cpp
flag_ptr = flags(from_rank).reduce_ptr(tile_idx);
wait_eq_sys(flag_ptr);
wait_eq_dev(flags(rank).epilogue_ptr(tile_idx));
```

waitsys就是跨peer，waitdev就是gpu内。

### gemm template对自定义epi的声明

```cpp
cutlass::gemm::kernel::GemmkWithVisitor<..., decltype(params.evt()), ...>
```

相当于告诉gemm模版这里有个visitor，需要经过一遍。

### gemmrs版本的swizzle如何编排

swizzle决定了cta-tile的调度顺序。也就是怎么把blocktile指派给不同rank。这个是防止所有rank都往一个rank的某个同一chunk写引发性能问题。
swizzle的编排，如何实现？

普通矩阵乘法是把AB=C[M,N]的M和N切成tileM和tileN，每个cta做一个tileC[tileM,tileN]。swizzle就是把不同的tileC指派给不同cta的调度方案。

gemmrs专属的特点是，算出来的tileC要被切成N个chunk，其中N-1个chunk要交给其他rank持有。本地只留1/N。如果每个rank都先算属于自己的这一块chunk，后算其他人的chunk，就会导致所有rank同时通信同时计算，达不到掩盖的目的。最好的做法是先算remote chunk，再算local chunk。但是不代表一个rs的chunk shape就和tileC相同，只是给rs的通信粒度切到了和tileC一样。

flux原生实现里面，也是刚好用一个tileC当一个rs的通信粒度，再次强调不代表语义上的chunk shape等于tileC shape：

```cpp
bytedance::flux::ReduceScatterOp<T, ThreadblockShape::kM, ThreadblockShape::kN, kFlattenTile>rs_op;
```

如果是八卡，怎么确定算哪一个remote chunk呢。flux里面是hardcode了一个表：

```cpp
#pragma once

namespace bytedance::flux {
constexpr static int kLocalWorldSize = 8;
constexpr static int kStages = 4;
struct Topology {
  int rank_from[4][8];
  int rank_to[4][8];
  int unused_segments_push[8];
  int segments[4][2];
  int rank_index[2][8];
};
/*
  ring mode: topo 0
  1rd stage: 4 -> [0 -> 1 -> 2 -> 3] -> [7 -> 6 -> 5 -> 4] -> 0
  2rd stage: 5 -> [1 -> 2 -> 3 -> 0] -> [4 -> 7 -> 6 -> 5] -> 1
  3nd stage: 6 -> [2 -> 3 -> 0 -> 1] -> [5 -> 4 -> 7 -> 6] -> 2
  4st stage: 7 -> [3 -> 0 -> 1 -> 2] -> [6 -> 5 -> 4 -> 7] -> 3

  no ring mode: topo 1
  1rd stage: 4 -> [0 -> 1 -> 2 -> 3] -> [7 -> 6 -> 5 -> 4] -> 0
  2rd stage: 5 -> [1 -> 0 -> 3 -> 2] -> [6 -> 7 -> 4 -> 5] -> 1
  3nd stage: 6 -> [2 -> 3 -> 0 -> 1] -> [5 -> 4 -> 7 -> 6] -> 2
  4st stage: 7 -> [3 -> 2 -> 1 -> 0] -> [4 -> 5 -> 6 -> 7] -> 3

*/
constexpr static __device__ Topology kTopologys[] = {
    // topo 0
    {{{4, 0, 1, 2, 5, 6, 7, 3},
      {3, 5, 1, 2, 0, 6, 7, 4},
      {3, 0, 6, 2, 5, 1, 7, 4},
      {3, 0, 1, 7, 5, 6, 2, 4}},
     {{1, 2, 3, 7, 0, 4, 5, 6},
      {4, 2, 3, 0, 7, 1, 5, 6},
      {1, 5, 3, 0, 7, 4, 2, 6},
      {1, 2, 6, 0, 7, 4, 5, 3}},
     {3, 0, 1, 2, 5, 6, 7, 4},
     {{3, 4}, {0, 5}, {1, 6}, {2, 7}},
     {
         {7, 3, 4, 0, 5, 1, 6, 2},  // numa node 0
         {0, 4, 1, 5, 2, 6, 3, 7},  // numa node 1
     }},
    // topo 1
    {{{4, 0, 1, 2, 5, 6, 7, 3},
      {1, 5, 3, 0, 7, 4, 2, 6},
      {3, 0, 6, 2, 5, 1, 7, 4},
      {1, 2, 3, 7, 0, 4, 5, 6}},
     {{1, 2, 3, 7, 0, 4, 5, 6},
      {3, 0, 6, 2, 5, 1, 7, 4},
      {1, 5, 3, 0, 7, 4, 2, 6},
      {4, 0, 1, 2, 5, 6, 7, 3}},
     {3, 2, 1, 0, 7, 6, 5, 4},
     {{3, 4}, {2, 5}, {1, 6}, {0, 7}},
     {
         {7, 3, 6, 2, 5, 1, 4, 0},
         {0, 4, 1, 5, 2, 6, 3, 7},
     }}};
}  // namespace bytedance::flux

```

感觉在写编译器......

### 2D Ring优化

2D Ring就是说1D Ring的链路不能总打平，分成两层，inter和intra，只在层内做ring。

同numa下通信开销小，跨numa/inter通信开销大。既然是2D ring，那么两个维度的ring的部分和就要用不同的指针。
local用reduce_ptr存还没加完的部分和，inter用reduce_sub_node_ptr存部分和，用一个flag表示是否写完。

关于1D和2D版本的部分和保存我画了张图：

![1D 和 2D Ring 部分和保存示意](/images/llm-system-comm-comp-fusion-operators/flux-1d-2d-ring-partial-sum.png)

## 自己的类flux实现

### test case

跑了一下发现自己的flux测出来效果远低于flux原生，甚至低于cutlass+nccl的baseline。跑了十组case发现自己测出来的时间非常不稳，而且和size关系不大。因此看起来是流同步一类的开销。另外还有个开销是要保证reduce的buffer一开始是清零状态，非0的话reduce操作会出错。

所以设计了这样一个消融方案去测：
因为一次iter可以这么拆：

```text
  A = cudaMemsetAsync(output)
  B = cudaStreamSynchronize after clear
  C = host thread barrier after clear
  D = launch fused GEMMRS kernel
  E = cudaStreamSynchronize after kernel
  F = host thread barrier after kernel
```

其中ABC就是我说的清零clear操作。
然后消融测试：

```text
  kernel-batched      = D D D D ... 
  kernel-sync         = (D + E + F) ...
  clear-only          = (A + B + C) ...
  clear-kernel-sync   = (A + B + C + D + E + F) ...
```

然后就很好测每个阶段都多少时间了。
最后测出来发现主要还是流同步那边花时间多，clear操作是几十us，占整个算子的不到10%。而同步时间开销比算子本身计算的时间开销高一倍以上。
但是不代表就没事了，如果清零之后不做同步，gemmrs算子就不能开始。如果做了同步就会很慢。这里要做的除了stream级别的clear-gemmrs保序，还有cross rank的同步。不可能一个rank已经开始gemm了，另一个rank还在清零。所以必须得等所有rank做完以后host这边做一个收口，确定都zero了再去gemmrs。这种比较普通的思路，开销不可接受，所以就研究了一下flux怎么实现的这部分。

### device side barrier

我实现的是clear kernel-host sync-gemmrs kernel，这个需要在host中转一次。flux里面把这个控制下放到了kernel里面做，理论上抖动会更小（不受cpu影响）而且更快。这个barrier kernel基于cudaipc/nvshmem做的。实现不是很长：

```cpp
__global__ void
CudaIpcBarrierAllKernel(CudaIpcBarrierAllArgs args) {
  int **sync_buffers = args.sync_buffers;
  int world_size = args.world_size;
  int cur_rank = args.rank;
  if (threadIdx.x < world_size) {
    __threadfence_system();
    // set achieved flag for others
    int *sync_buffer_dst = sync_buffers[threadIdx.x] + cur_rank;
#pragma unroll 1
    while (atomicCAS_system(sync_buffer_dst, 0, 1) != 0) {
    }
    __threadfence_system();
    int *wait_ptr = sync_buffers[cur_rank] + threadIdx.x;
#pragma unroll 1
    while (atomicCAS_system(wait_ptr, 1, 0) != 1) {
    }
    __threadfence_system();
  }
}
```

这个是alltoallbarrier，也有ringbarrier。注意这种barrier不能用在cta比较多的情况，因为逻辑上的cta如果很多，某些逻辑cta已经走到了barrier点占着物理sm不释放，有的逻辑cta还没有被调度上去，就无尽死锁了。
