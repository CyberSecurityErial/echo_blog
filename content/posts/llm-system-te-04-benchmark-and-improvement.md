---
date: '2026-05-21T06:40:00+08:00'
draft: false
title: 'LLM System: Transformer Engine 04 - Benchmark、Trace 与改进方向'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Transformer Engine", "Benchmark", "Profiler"]
series: ["LLM System", "Transformer Engine"]
series_order: 4
weight: 5
math: true
---

> 本篇目标：从会用 TE API 走到能 benchmark、trace 和分析 TE 的性能路径。

这篇先记录实验路线。学习 TE 不能只读 API，最后一定要落到 profiler trace、kernel 数量、访存、Tensor Core utilization 和通信 overlap 上。

## 0. 学习方法

按四步推进：

```text
1. 阅读：先知道 TE module 替换了什么
2. 实验：写最小 benchmark 对比 torch vs TE
3. Trace：用 nsys / ncu 看 kernel 和通信
4. 小改动：从 docs、examples、benchmark 工具开始贡献
```

## 1. 模块替换实验

先做最小模块替换实验，确认 forward / backward 数值基本一致，再看性能差异。

对比路径：

```text
torch.nn.Linear       → te.Linear
torch.nn.LayerNorm    → te.LayerNorm / te.RMSNorm
LN + Linear           → te.LayerNormLinear
LN + MLP              → te.LayerNormMLP
Attention             → te.MultiheadAttention
Transformer block     → te.TransformerLayer
```

## 2. FP8 recipe 实验

对比 BF16 / FP16 / FP8，追踪 `amax`、`scale`、`fp8_format`、`amax_history_len`。这一步的重点不是只看吞吐，而是理解 scale 策略如何影响数值稳定性。

## 3. Profiler 对比

先做三组 profiler 对比：

三组实验：

```text
torch LayerNorm + Linear
vs
te.LayerNormLinear

torch MLP
vs
te.LayerNormMLP

PyTorch attention / SDPA
vs
TE MultiheadAttention
```

## 4. Microbenchmark 工具

可以单独维护一个 `te_bench/` 小工具，目标是自动对比 torch / TE 在不同 shape、dtype、FP8 recipe 下的性能。

计划目录：

```text
te_bench/
├── bench_linear.py
├── bench_layernorm_linear.py
├── bench_mlp.py
├── bench_attention.py
├── bench_transformer_layer.py
└── parse_nsys.py
```

## 5. 改进方向

优先级先这样排：

```text
1. examples / docs
2. benchmark / profiler 工具
3. Megatron + TE config / checkpoint / recompute 适配理解
4. MoE GroupedLinear 性能分析
5. 通信重叠和 sm_margin 深挖
```

## 6. 关键链接

```text
Transformer Engine GitHub:
https://github.com/NVIDIA/TransformerEngine

Transformer Engine examples:
https://github.com/NVIDIA/TransformerEngine/tree/main/examples

Transformer Engine PyTorch API:
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html

NVIDIA Nsight Systems:
https://developer.nvidia.com/nsight-systems

NVIDIA Nsight Compute:
https://developer.nvidia.com/nsight-compute
```
