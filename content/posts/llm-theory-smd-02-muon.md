---
date: '2026-05-17T17:56:11+08:00'
draft: true
title: 'LLM Theory: SMD 02 - Muon 优化器的球面运动视角'
categories: ["LLM Theory"]
tags: ["LLM", "LLM Theory", "Optimization", "SMD", "Muon"]
series: ["LLM Theory", "SMD"]
series_order: 2
weight: 4
math: true
---

> 本文是 LLM Theory 下 SMD 专题的第二篇，延续上一篇对 SGD、Adam、AdamW 的径向/切向分解，准备用同一套视角分析 Muon 优化器。

## 想回答的问题

1. Muon 的 update 和普通 SGD / AdamW update 的核心差别是什么？
2. 如果把 Muon 的更新量拆成径向分量和切向分量，主要改变的是半径还是方向？
3. Muon 的正交化/归一化操作，在 SMD 视角下对应什么训练动力学？
4. 它和 AdamW 的 weight decay、有效学习率之间是什么关系？

## 记号

TODO

## Muon 的更新式

TODO

## 用 SMD 拆解 Muon

TODO

## 和 AdamW 的对比

TODO

## 参考文献

TODO
