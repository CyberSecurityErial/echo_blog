---
date: '2026-05-21T06:00:00+08:00'
draft: false
title: 'LLM System: Transformer Engine 00 - 学习地图'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Transformer Engine", "FP8", "Megatron"]
series: ["LLM System", "Transformer Engine"]
series_order: 0
weight: 1
math: true
---

> 这篇文章是 LLM System 系列里 Transformer Engine 子专题的第 0 篇，也是这个主题的学习入口。

我准备用这个系列系统学习 [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)。目标不是只会调用几个 `te.*` API，而是把 TE 放到大模型训练系统里理解：它为什么存在、如何利用 FP8 / FP4 和 fused kernel、怎么被 Megatron-LM 接入，以及后续如何用 profiler 分析和改进它。

## 0. 学习目标

这组笔记会围绕四件事展开：

```text
1. TE 在 AI Infra 技术栈中的位置
2. TE 的核心加速策略：低精度、融合算子、Attention、MoE、通信重叠
3. Megatron-LM / Megatron-Core 如何接入 TE
4. 如何 benchmark、trace 并尝试改进 TE
```

## 1. 技术定位

Transformer Engine 不是训练框架，而是 NVIDIA 为 Transformer 训练/推理提供的高性能 building block 库。它大致位于训练框架和底层 CUDA/cuBLAS/cuDNN kernel 之间。

这一阶段要回答的问题：

```text
1. TE 和 PyTorch AMP 的关系是什么？
2. TE 和 Megatron-LM 的边界在哪里？
3. TE 为什么不是一个完整训练框架？
4. TE 为什么不只是 FP8，而是 Transformer 优化库？
5. TE 在 AI Infra 技术栈里更靠近 compiler/runtime/kernel，还是更靠近 model framework？
```

## 2. 核心加速策略

核心加速策略先按五条线学习：

```text
1. FP8 / MXFP8 / NVFP4
2. fused Linear / Norm / MLP
3. fused attention / layout / backend
4. MoE Grouped GEMM
5. 通信重叠 / sm_margin / UserBuffer
```

## 3. 框架耦合

重点记录 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) / Megatron-Core 如何接入 Transformer Engine。这里不是简单替换模块，而是会涉及 `TransformerConfig`、TP/SP/CP/EP 参数、FP8 recipe、activation checkpoint、recompute 和 quantized tensor 保存策略。

## 4. 上手和改进

上手顺序按阅读、实验、trace、小改动推进。先能替换 `torch.nn.Linear` / `torch.nn.LayerNorm`，再看 `LayerNormLinear`、`LayerNormMLP`、`MultiheadAttention`、`TransformerLayer`，最后用 `nsys` / `ncu` 分析 kernel 和通信。

## 5. 系列文章

```text
1. Transformer Engine 在 AI Infra 技术栈中的位置
2. Transformer Engine 的核心加速策略：FP8、融合算子与 Attention
3. Megatron-LM 如何接入 Transformer Engine
4. 如何 benchmark、分析并改进 Transformer Engine
```

## 6. 关键链接

```text
Transformer Engine GitHub:
https://github.com/NVIDIA/TransformerEngine

Transformer Engine 文档:
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html

Transformer Engine PyTorch API:
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html

Megatron-LM GitHub:
https://github.com/NVIDIA/Megatron-LM

Megatron-Core TE extension 文档:
https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.extensions.transformer_engine.html
```
