---
date: '2026-05-21T10:30:00+08:00'
draft: false
title: 'LLM System: Transformer Engine 03 - Megatron-LM 如何接入 TE'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Transformer Engine", "Megatron", "Distributed Training"]
series: ["LLM System", "Transformer Engine"]
series_order: 3
weight: 4
math: true
---

> 本篇目标：搞清楚 Megatron-LM / Megatron-Core 如何把 Transformer Engine 接入真实训练系统。

这篇先记录阅读入口和问题清单。TE 单独看是一个高性能库，但它真正的价值通常是在 Megatron-LM / Megatron-Core / NeMo 这类训练系统里释放出来。

## 0. 二者分工

可以先用一句话区分：

```text
Megatron = 训练系统总控
TE       = Transformer block 高性能执行后端
```

Megatron 负责 TP / PP / DP / SP / CP / EP、训练 loop、optimizer、checkpoint、activation checkpoint、MoE routing 和 pipeline schedule。TE 负责 Transformer layer 内部的 FP8 / FP4 recipe、fused Linear / Norm / MLP / Attention、quantized tensor 管理，以及一部分和并行训练相关的模块适配。

## 1. Megatron 中 TE 的入口

第一步先从 Megatron 的 TE extension 层看起。

需要追踪的关键词：

```text
--transformer-impl transformer_engine
fp8
fp8_format
fp8_recipe
transformer_config
TransformerLayer spec
TENorm
TELinear
TEDotProductAttention
TEColumnParallelLinear
TERowParallelLinear
```

## 2. TransformerConfig 如何传给 TE

重点看 Megatron 的 `TransformerConfig` 如何决定使用 local transformer impl 还是 `transformer_engine` impl，以及这些配置如何继续传进 TE module。

## 3. TP / SP / CP / EP 参数传递

TE 模块里出现 `sequence_parallel`、`tp_group`、`process_group` 这类参数，说明它并不是完全无状态的底层算子封装。后面要重点看这些并行组信息到底在哪里被消费。

## 4. FP8 checkpoint / recompute

重点关注 quantized tensor、activation 保存和 recompute 的边界。FP8 训练里保存什么 tensor、保存原始输入还是 quantized input，会直接影响显存、数值和重算策略。

## 5. Sequence Parallel

从 `sequence_parallel` 和 `return_layernorm_output_gathered` 理解为什么底层模块会暴露并行语义。这是 TE 和 Megatron 耦合很深的一个切入口。

## 6. 调用链图

先画一条粗粒度调用链：

```text
Megatron training loop
    ↓
Megatron model provider
    ↓
TransformerConfig
    ↓
TransformerLayer spec
    ↓
TE TransformerLayer / TE modules
    ↓
FP8 autocast / quantization recipe
    ↓
fused norm / fused GEMM / fused attention
    ↓
cuBLAS / cuDNN / CUDA kernel
```

## 7. 关键链接

```text
Megatron-LM GitHub:
https://github.com/NVIDIA/Megatron-LM

Megatron-Core Transformer Engine extension:
https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.extensions.transformer_engine.html

Megatron TE extension 源码入口:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/extensions/transformer_engine.py

Transformer Engine GitHub:
https://github.com/NVIDIA/TransformerEngine
```
