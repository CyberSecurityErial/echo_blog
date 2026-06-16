---
date: '2026-06-16T00:00:00+08:00'
draft: false
title: 'LLM System: 基础知识速查 01 - MoE'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "MoE", "基础知识速查"]
series: ["LLM System", "基础知识速查"]
series_order: 1
weight: 1
math: true
---

## moe f&b 计算流程

### Router

Router 计算每个 token 应该路由到哪些 expert。

![MoE Router 计算 01](/echo_blog/images/llm-system-basic-quick-reference-moe/01-router-01.png)

![MoE Router 计算 02](/echo_blog/images/llm-system-basic-quick-reference-moe/02-router-02.png)

### Dispatch

Dispatch 根据 router 结果，把 token 分发到对应 expert 的输入 buffer。

![MoE Dispatch 计算 01](/echo_blog/images/llm-system-basic-quick-reference-moe/03-dispatch-01.png)

![MoE Dispatch 计算 02](/echo_blog/images/llm-system-basic-quick-reference-moe/04-dispatch-02.png)

### FFN

每个 expert 内部执行自己的 FFN / MLP 计算。

![MoE FFN 计算 01](/echo_blog/images/llm-system-basic-quick-reference-moe/05-ffn-01.png)

![MoE FFN 计算 02](/echo_blog/images/llm-system-basic-quick-reference-moe/06-ffn-02.png)

### Combine

Combine 把各 expert 的输出按路由权重聚合，并还原到 token 维度。

![MoE Combine 计算 01](/echo_blog/images/llm-system-basic-quick-reference-moe/07-combine-01.png)

![MoE Combine 计算 02](/echo_blog/images/llm-system-basic-quick-reference-moe/08-combine-02.png)
