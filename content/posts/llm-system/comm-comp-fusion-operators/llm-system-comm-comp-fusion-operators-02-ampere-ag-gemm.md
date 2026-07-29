---
date: '2026-07-18T00:00:00+08:00'
draft: true
title: 'LLM System: 通信计算融合算子实现 02 - sm80 AllGather + GEMM'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "CUDA", "GEMM", "AllGather", "Communication Overlap", "Distributed Training"]
series: ["LLM System", "通信计算融合算子实现"]
series_order: 2
weight: 2
math: true
---

## flux实现：sm80 ag+gemm
先介绍基本概况。

基本输入情况是，一个矩阵A沿着M切了8片（其实不一定是8不过tp一般都开8，这里直接hardcode一下），每片是[M/8, K]分散在8个rank。每个rank上面的B矩阵是完整的[K, N]，最后要计算的是完整的A和本地的B的乘法结果，发现把A按行切是不会让分片间有数据依赖的，自然而然会想到做一个流水，哪个分片先到就让哪个分片先gemm。做gemm的cta算一下自己的分片对应的物理tile序号是哪个，然后根据这个序号找到地址直接取数即可。但是为了保证gemm的效率，不代表gemm时候的那个tile要和M/8相等，gemm tileM是M/8的1/2或者1/4可能是比较好的，如果大于M/8会很糟糕，因为得等好多个通信的分块才能算一次gemm。

## 引入stream-K
一般来说让一个cta负责一个tilem*tileN，不在K维度切。如果K很大，为了打满并行，K维度也要切开并且分给很多cta同时做，做完了以后有一个reduce。这个过程有bitwise问题不过不是这里讨论的问题。

在没有stream-K的时候swizzle操作只是简单的让物理tile编号和逻辑tile编号基于集合通信shard到达的顺序

