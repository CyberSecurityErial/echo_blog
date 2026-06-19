---
date: '2026-06-17T00:00:00+08:00'
draft: false
title: 'LLM System: 训练框架随笔 04 - Megatron 通信器设计：如何防死锁'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Training Framework", "Megatron", "Communication", "NCCL"]
series: ["LLM System", "Training Framework Notes"]
series_order: 4
weight: 1
math: true
---

> 本篇目标：

## 为什么通信器会死锁

## Megatron 的 P2P 通信抽象

## batch p2p comm

## overlap p2p comm

## warmup / steady / cooldown 里的通信顺序

## 如何设计防死锁的通信接口

## TODO

