---
date: '2026-07-04T19:19:40+08:00'
draft: true
title: 'LLM System: 算法和 Infra 交织的 RL 杂谈 02 - RL 训推误差在 System 层面的本质和解决（容忍）'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "RL", "AI Infra"]
series: ["LLM System", "算法和 Infra 交织的 RL 杂谈"]
series_order: 2
weight: 2
math: true
---


## system训推误差的本质

字节有一篇论文说了，除了看logprob的diff其他的评估训推一致性