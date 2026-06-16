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

### MoE softmax
MoE做router的意义是什么？

MoE 做softmax的意义和一般softmax意义类似，都是让logits能被解释为概率。
被解释为概率/加权的几个要求：正数，和为1，尺度一致，

哪里用了softmax？

Router阶段：router求r的时候，可以先做softmax后做topk，也可以先做topk后做softmax。先做topk那就是logits（expert）之间做筛选，后做topk那就是根据加权筛选。

或者二者结合，先做softmax，然后topk，再在被选中的expert内部做一次topk



