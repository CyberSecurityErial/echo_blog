---
date: '2026-05-21T06:20:00+08:00'
draft: true
title: 'LLM System: Transformer Engine 02 - 核心加速策略'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Transformer Engine", "FP8", "Attention", "MoE"]
series: ["LLM System", "Transformer Engine"]
series_order: 2
weight: 3
math: true
---

> 本篇目标：理解 Transformer Engine 的核心加速策略，包括低精度、融合算子、Attention、MoE Grouped GEMM 和通信重叠。

这篇先作为学习提纲。后面读源码和跑实验时，会逐步把每条线补成完整笔记。

## 0. 总览

TE 的性能收益大致来自三类地方：

```text
1. 低精度：FP8 / FP4 降低计算和访存压力
2. 融合算子：减少中间 tensor 写回 HBM 和 kernel launch
3. 训练系统信息：结合 TP / SP / MoE / 通信重叠选择更合适的路径
```

## 1. 低精度：FP8 / MXFP8 / NVFP4

这一条线先围绕 FP8 建立直觉，再继续看 Blackwell 上的 MXFP8 / NVFP4。

重点概念：

```text
FP8 E4M3
FP8 E5M2
HYBRID recipe
amax
scale
inverse scale
delayed scaling
current scaling
block scaling
FP8 tensor cache
```

量化和反量化关系可以先粗略理解成：

\[
x_{\mathrm{fp8}} = \operatorname{round}\left(\frac{x}{s}\right)
\]

\[
x \approx s \cdot x_{\mathrm{fp8}}
\]

## 2. 融合算子：Norm / Linear / MLP

先从 `torch.nn.LayerNorm + torch.nn.Linear` 和 `te.LayerNormLinear` 的对比入手。普通写法会产生中间 `normalized_x`，TE fused path 的直觉是减少中间 tensor 读写和 kernel launch。

## 3. Attention：layout / mask / backend

Attention 部分重点看 `DotProductAttention` / `MultiheadAttention`，以及 QKV layout、mask、GQA / MQA 和 backend 选择。

重点问题：

```text
1. TE 的 DotProductAttention / MultiheadAttention 和 PyTorch SDPA 有什么关系？
2. TE attention 支持哪些 QKV layout？
3. causal mask、padding mask、packed sequence 怎么处理？
4. GQA / MQA 支持路径在哪里？
5. fused attention backend 如何选择？
6. 和 FlashAttention、cuDNN attention 的边界在哪里？
```

## 4. MoE / Grouped GEMM

这部分结合 Megatron-Core 的 MoE 实现看 `GroupedLinear`、`Grouped GEMM`、`moe_permute`、`moe_unpermute`。核心问题是：多个 expert 的小 GEMM 如何打包，token permutation / unpermutation 的开销如何影响整体收益。

## 5. 通信重叠

这部分重点理解 `sm_margin`、`sequence_parallel`、`UserBuffer` 和 communication overlap。TE 不是纯算子库，它的一些参数已经带有大规模训练系统的并行语义。

## 6. 关键链接

```text
Transformer Engine GitHub:
https://github.com/NVIDIA/TransformerEngine

FP8 / FP4 primer:
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html

PyTorch API:
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html

Fused attention C API:
https://nvidia.github.io/TransformerEngine/api/c/fused_attn.html
```
