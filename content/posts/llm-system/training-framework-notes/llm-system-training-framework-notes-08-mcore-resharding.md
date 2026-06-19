---
date: '2026-06-19T00:00:00+08:00'
draft: false
title: 'LLM System: 训练框架随笔 08 - MCore Checkpoint Resharding'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Training Framework", "Megatron", "MCore", "Resharding"]
series: ["LLM System", "Training Framework Notes"]
series_order: 8
weight: 1
math: true
---

## 简述
resharding这个词出现了好几次，rl的resharding，ckpt的resharding，fsdp的resharding，这部分先看ckpt的resharding目的是什么，怎么做的。

ckpt的resharding发生在训练开始前loadckpt的时候，ckpt的格式有三种，torchdist，torchdcp，fsdpdtensor。torchdist是mcore原生的distckpt，torchdcp是torchdist的原生格式，fsdp是fsdp2的dcp格式，参数是dtensor分片。

加载ckpt的时候sharded_state_dict_default演算一个ckpt shard在全局唯一确定的位置。
输入的信息：data，shape，tpaxis，tprank，tpsize，layerkey（name），replicaid
输出的信息：key，data，localshape，globalshape，globaloffset，axisinfo

这里ckpt的resharding做的事情是当我们load一个ckpt，这个ckpt里面包含它被保存时候的并行信息，但是它被保存的时候的并行设置不一定和这次重新load的时候一样。所以这个resharding就是一次覆盖。那么能否直接不给ckpt加入并行信息呢，因为之前就是分布式的ckpt，没有之前的信息无法拼成完整的模型。

## 异步分布式ckpt的难点
训练的主路径不想等io，所以ckpt的过程是async的，但是ckpt的各个shard还需要同步。mcore和torchdcp在这里权衡的方式是在快照的时候做同步，但是分批写入的时候是异步的。也就是每个step的边界快照一次，然后异步并行传shard。另外同时最多只限制一个async的异步shard组在inflight，否则很难debug。那么如果一批全局async的shard还没都写入，其他新step的同步快照ckpt就得一直等着。所以这个地方也得开足够的空间保存这些shard，这个地方可以pinCPUmem。
