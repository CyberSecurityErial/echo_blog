---
date: '2026-08-17T00:00:00+08:00'
draft: false
title: 'gemm和alltoall通算融合'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "CUDA", "GEMM", "AllToAll", "Ulysses", "GQA", "Communication Overlap"]
series: ["LLM System", "通信计算融合算子实现"]
series_order: 3
weight: 3
math: true
---

## 1. 总体思想

这次做的是单机八卡 H200、NVLink、Ulysses CP 下的 GEMM 和 AllToAll 融合。先把 forward 写清楚：

```text
A2A → QKV projection → QK → PV → A2A
```

需要接起来的主边界有两个。输入侧是 `A2A→QKV projection`，通信先把各个 peer 的输入 tile 搬到本地最终布局，GEMM 拿到一块就算一块。输出侧是 `batched PV→A2A`，每个本地 head 都有一组独立的 `P×V`，GEMM 算完一个 tile，通信 CTA 立刻把它送到目标 rank 的最终 Ulysses 布局。

如果 GEMM 和 NCCL 顺序执行，端到端时间接近两段时间相加。这里把通信 CTA 和 GEMM CTA 放进同一个 cooperative persistent grid，两种 CTA 常驻在不同的 SM 上，用 tile 级 ready epoch 接力。`A2A→GEMM` 由通信生产、GEMM 消费；`GEMM→A2A` 交换生产消费关系。这样首批 tile 到达后就能启动计算，前面的 tile 也可以在后续 GEMM 还在跑时发出去。

![persistent grid 内通信 CTA 与 GEMM CTA 的 tile 流水](/images/llm-system-comm-comp-fusion-operators/persistent-cta-tile-pipeline.svg)

## 2. 优化的具体方法

GEMM mainloop 直接复用 CUTLASS 3.x 的 Hopper TMA/WGMMA collective。BF16 projection 用 `128×256×64`、Cooperative、stage 4、cluster `(2,1,1)`；BF16 batched PV 用 `128×128×64` Pingpong；FP8 projection 用 `128×128×128` Cooperative FastAccum。三种 shape 分开选 policy，共用 persistent 调度、epoch 和 Ulysses route。

输入方向的 ready 粒度细到 `[batch, M tile, peer/K group]`。通信 CTA 按 `[window, peer, M tile, row group]` 生产数据，M window 根据当前 GEMM 工作前沿动态算，范围限制在 1 到 8。GEMM 只在 peer/K group 边界做一次 `acquire.sys`，等待只落在负责发 TMA 的线程上，跨 group 时提前到下一次 pipeline acquire 之前，MMA 内循环不做反复轮询。BF16 通信 stage 取 64 行，FP8 取 32 行；尾块走同一个 monolithic kernel 里的 vector 路径。

输出方向让 CUTLASS epilogue 正常把 D tile 写回本地显存，等 TMA store 完成后只做一次很小的 `release.sys`。通信 CTA 看到 epoch 后直接写入对端最终布局。Q、K、V 三段会按 `Hq/Hkv/world` 各自切分，GQA 的 K/V 不需要复制，也没有额外的完整 pack tensor。

ready flag 放在 global memory，值是单调递增的 epoch。每轮 benchmark 不清零 flag，省掉 reset kernel，也避开布尔 flag 跨轮复用的问题。通信 CTA 数量通过 sweep 找 GEMM 损失和 NVLink 带宽的交点，最终默认值是：BF16 medium projection 12，BF16 large projection 10，FP8 输入 projection 8，batched PV 32。附加的 FP8 QKV 输出 route 使用 40 个通信 CTA。

projection 的 cluster `(2,1,1)` 会把两个 CTA 组成一个 cluster，实际用到了 TMA multicast。通信区和计算区都按完整 cluster 对齐，launch 前用真实 block、shared memory 和 cluster shape 检查整张 grid 能否同时驻留，避免某些 CTA 占住 SM 后互相等死。和 cluster 1 相比，medium shape 快了约 `0.8%`，large shape 基本持平。

最后跑了 CP2/4/8、M tail、非 TileK 对齐的 peer shard、GQA route、forward/inverse route 和连续 8 个 epoch。完整 memcheck 是 `0 errors`，racecheck 是 `0 hazards / 0 errors / 0 warnings`。

## 3. 最终完整的数据结果

所有正式结果都是 10 次 warmup 加 50 次测量，每个样本取所有 rank 中最慢的 CUDA event 时间。CP4 使用物理 GPU `0,2,4,5`，CP8 使用全部八张卡。吞吐统一按 GEMM 的 `2LMNK` 计算，通信本身不增加 FLOPs，但表里的端到端时间包含完整通信，所以这个数字能直接看出 AllToAll 吃掉了多少 GEMM 吞吐。

下面每个百分比都以同一行最快的 pure GEMM 为 `100%`：

| 主融合边界 / dtype / CP / `(M,N,K,L)` | fused：延迟 / 每卡 TFLOPS（占 pure） | fastest pure GEMM：延迟 / 每卡 TFLOPS | TE 或标准 GEMM+NCCL：延迟 / 每卡 TFLOPS（占 pure） | Flux 1.1.2：延迟 / 每卡 TFLOPS（占 pure） |
|---|---:|---:|---:|---:|
| A2A→projection，BF16，CP4，`(2048,5120,4096,1)` | **0.1470 ms / 584.4（76.6%）** | cuBLASLt 0.1126 ms / 763.0 | TE+NCCL 0.3794 ms / 226.4（29.7%）；cuBLAS+NCCL 0.2596 ms / 330.9（43.4%） | 0.2319 ms / 370.5（48.6%） |
| A2A→projection，BF16，CP8，`(2048,5120,4096,1)` | **0.1888 ms / 455.0（78.6%）** | cuBLASLt 0.1484 ms / 578.9 | TE+NCCL 0.3059 ms / 280.8（48.5%）；cuBLAS+NCCL 0.2840 ms / 302.5（52.3%） | 0.2655 ms / 323.6（55.9%） |
| A2A→projection，BF16，CP8，`(4096,10240,8192,1)` | **1.1468 ms / 599.2（88.5%）** | cuBLAS 1.0155 ms / 676.7 | TE+NCCL 1.3941 ms / 492.9（72.8%）；cuBLAS+NCCL 1.3871 ms / 495.4（73.2%） | 1.2459 ms / 551.6（81.5%） |
| A2A→projection，FP8，CP8，`(4096,10240,8192,1)` | **0.6602 ms / 1041.0（85.6%）** | cuBLASLt 0.5649 ms / 1216.6 | cuBLASLt FP8+NCCL 0.8115 ms / 846.8（69.6%） | 融合算子只支持 BF16，FP8 构造器运行时拒绝 |
| batched PV→A2A，BF16，CP8，`(4096,128,4096,8)` | **0.1129 ms / 304.5（75.4%）** | CUTLASS 0.0851 ms / 403.9 | cuBLAS batched GEMM+NCCL 0.2408 ms / 142.7（35.3%；按其自身 pure 为 45.5%） | 没有等价的 strided-batched PV 融合算子 |

还有一条附加测试，用来检查输出 route 的泛化能力。它做 `dense QKV projection→GQA-pack A2A`，精确对照是 `torch._scaled_mm` 落到 cuBLASLt，再把 Q/K/V 三段按 GQA head 切开做 NCCL A2A，计算顺序和最终布局都一致。这条不在上面的主 forward 边界里：

| 附加测试 / dtype / CP / `(M,N,K,L)` | fused：延迟 / 每卡 TFLOPS（占 pure） | fastest pure GEMM：延迟 / 每卡 TFLOPS | 精确 GEMM+NCCL：延迟 / 每卡 TFLOPS（占 pure） | Flux 1.1.2 |
|---|---:|---:|---:|---:|
| dense QKV projection→GQA-pack A2A，FP8，CP8，`(4096,10240,8192,1)` | **0.8675 ms / 792.1（61.7%）** | CUTLASS 0.5356 ms / 1283.0 | cuBLASLt FP8+NCCL 1.2260 ms / 560.5（43.7%；按其自身 pure 为 46.4%） | 融合算子只支持 BF16，FP8 构造器运行时拒绝 |

H200 dense Tensor Core 的标称峰值按 BF16 `989.5 TFLOPS/GPU`、FP8 `1979 TFLOPS/GPU` 计算。由于分母包含 A2A，下面的比例表示完整融合边界相对硬件 GEMM 峰值还能留下多少吞吐：

| dtype / large projection | H200 标称峰值 | fused 端到端吞吐 | 占硬件峰值 |
|---|---:|---:|---:|
| BF16 A2A→projection | 989.5 TFLOPS/GPU | 599.2 TFLOPS/GPU | 60.6% |
| FP8 A2A→projection | 1979 TFLOPS/GPU | 1041.0 TFLOPS/GPU | 52.6% |

本机单张快卡的 BF16 pure GEMM 实测目标是 `848.3 TFLOPS/GPU`。CP8 结果按最慢 rank 收口，其中三张卡锁在 1.5 GHz，表里仍然保留这个严格口径。从最终结果看，large BF16 已经留下 pure GEMM 的 `88.5%`，FP8 留下 `85.6%`；batched PV 这种计算更薄、通信占比更高的 shape，融合后是 pure 的 `75.4%`，顺序的 cuBLAS batched GEMM+NCCL 只有最快 pure 的 `35.3%`。
