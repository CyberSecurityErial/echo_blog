---
date: '2026-06-27T00:00:00+08:00'
draft: true
title: 'LLM System: RL 训推一致 02 - VeRL 的框架层训推一致处理'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "RL", "RLHF", "Train-Inference Consistency", "VeRL", "AI Infra"]
series: ["LLM System", "RL 训推一致"]
series_order: 2
weight: 2
math: true
---

# LLM System: RL 训推一致 02 - VeRL 的框架层训推一致处理

## 概述
VeXact主要保证算子级别的一致性，而VeRL的scope主要限于整个RL流程。工程上的说法是，VeRL本身管理：train-rollout-update循环里面各个环节中传递内容的版本同步问题。而理论一些的说法是，尽量让rollout和采样分布和train的采样分布尽可能一致。

## 一致性语义约束
说白了就是RL的业务背景，比如PPO的token必须mask掉非环境因素。这部分只需要熟悉RL算法本身即可，不涉及系统优化。
## 一致性非语义约束
这部分是RL计算中语义体现不出来或不刻意强调的约束。或者说纯算法业务之外，遇到算法系统耦合的地方都是这部分。

train actor对rollout engine做权重同步，版本一致。
train actor默认不使用rollout的logprob，自己重算。
如果要用rollout的logprob，框架提供importance sampling和rejection sampling做补偿train和rollout的logprob差异。
约束MoE Router路径。
