---
date: '2026-05-21T10:10:00+08:00'
draft: false
title: 'LLM System: Transformer Engine 01 - 在 AI Infra 技术栈中的位置'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Transformer Engine", "AI Infra"]
series: ["LLM System", "Transformer Engine"]
series_order: 1
weight: 2
math: true
---

> 本篇目标：了解 Transformer Engine 的技术定位，搞清楚它为什么存在，以及它和 PyTorch、cuBLAS、Megatron-LM 的边界。



## 基本接口

Layer 类定义接口非常直接，最表层的使用方式就是把 `torch.nn` 模块替换成 `transformer_engine.pytorch` 模块。

普通 PyTorch 写法：

```python
self.linear = torch.nn.Linear(hidden_size, 4 * hidden_size)
```

TE 写法：

```python
import transformer_engine.pytorch as te

self.linear = te.Linear(hidden_size, 4 * hidden_size)
```

但这只是 API 表面。TE 真正有价值的地方在于：它不是只换一个 Python 类，而是把 Transformer 训练中常见的低精度、fused op、并行信息和 backend 选择打包进模块里。

FP8 量化上下文大概长这样：

```python
from transformer_engine.pytorch import fp8_autocast

with fp8_autocast(enabled=True):
    y = module(x)
```

进入 TE 的 FP8 上下文之后，TE 会围绕量化、反量化、fused path、tensor cache 和 backend 选择做一系列处理。相较于纯 PyTorch 计算图优化，TE 会拿到一些更贴近训练系统的信息，例如 tensor parallel、sequence parallel、FP8 recipe 等。这些额外信息给底层算子优化留下了空间。

这一点目前还是比较 general 的认知，后面要继续顺着源码和 profiler trace 去验证。

## TE 和 Megatron 的边界

Megatron-LM / Megatron-Core 更像训练系统总控，负责模型并行、训练 loop、optimizer、activation checkpoint、MoE routing、pipeline schedule 和 config 系统。

TE 更像 Transformer block 的高性能执行后端，负责 `Linear`、`LayerNorm`、`RMSNorm`、`LayerNormLinear`、`LayerNormMLP`、`MultiheadAttention`、`TransformerLayer` 这些模块里的低精度和 fused kernel 路径。

后面第 03 篇会重点看 Megatron 里的 TE 接入层：

```text
megatron/core/extensions/transformer_engine.py
```

公开源码入口：

```text
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/extensions/transformer_engine.py
```

## TE 和 PyTorch 的边界

TE 会调用 PyTorch，也提供 PyTorch API，但它不是要替代整个 PyTorch。更准确地说，TE 只替换 Transformer 训练里最关键的一批模块，并在这些模块里接入 NVIDIA GPU 上更激进的低精度和 kernel 优化。

也就是说：

```text
PyTorch: 通用深度学习框架
TE: 面向 Transformer block 的高性能执行层
Megatron: 大规模训练系统和并行策略总控
```

## 关键链接

```text
Transformer Engine GitHub:
https://github.com/NVIDIA/TransformerEngine

Transformer Engine 文档:
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html

Transformer Engine PyTorch API:
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html

Megatron-LM GitHub:
https://github.com/NVIDIA/Megatron-LM
```
