---
date: '2026-06-27T00:00:00+08:00'
draft: true
title: 'LLM System: RL 训推一致 00 - 学术方案与工业框架调研'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "RL", "RLHF", "Train-Inference Consistency", "AI Infra"]
series: ["LLM System", "RL 训推一致"]
series_order: 0
weight: 0
math: true
---

# LLM System: RL 训推一致 00 - 学术方案与工业框架调研

## 什么是 RL 场景下的训推一致

浮点精度不满足加法交换律，分布式的并发原子操作高效的代价是无法保证操作顺序，导致了浮点误差，影响loss的计算。
这其中kernel做出的优化也包含在内，这些优化在某些切分角度也可以认为是做了无序并发的原子操作。
训推bitwise不一致来源于并发的原子操作，比如应用在Sequence Parallel的RingAttention。因此完全不可能100%避免。业界的方案是丢弃掉不可用的token（但如何定义不可用呢）。

## 论文调研

### 发现训推一致问题
Diagnosing Training Inference Mismatch in LLM Reinforcement Learning
这篇只是顺带一提讲了VeRL里面VeXact的训推一致同步，但是主要还是以算法分析为主。

### 解决训推一致的方案
* Deterministic Inference across Tensor Parallel Sizes That Eliminates Training–Inference Mismatch
TP一般都是TP=1 2 4 8这些，只在机内有，vllm也有custom allreduce适配这种高互联带宽小范围的tp。

所以对Tensor Parallel来说解决方案比较简单，硬编码1 2 4 8Rank时候allreduce的加法顺序即可。这里用的是树状加法，但是算子效率很低。

* VeRL-VeXact module
treeReduction顺序固定 + splitK顺序固定 + rollout/fsdp Kernel Align

### 训推一致的评价指标

1. training reward
2. validation reward
3. MoE gate deviation：衡量router选择路径的差异。只针对moe模型的metric。
4. gradient norm
5. 最直接的，rollout logits - fsdp logits

