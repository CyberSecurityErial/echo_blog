---
date: '2026-06-15T00:00:00+08:00'
draft: true
title: 'LLM System: 训练框架随笔 02 - Femtotron PP Schedule 模块重构日记'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Training Framework", "Pipeline Parallel", "Schedule", "Femtotron"]
series: ["LLM System", "Training Framework Notes"]
series_order: 2
weight: 1
math: true
---

> 本篇目标：

## 为什么要重构
因为要支持vpp（interleave 1f1b） zerobubble dualpipe dualpipev，之前的只支持1f1b，gpipe。
## 从vpp思索stage的执行序
在gpipe和1f1b里面，一个stage就是一个rank，但如果加上vpp，一个rank上就很多虚拟的stage了。以前设计vpp的初衷就是，一个mb在等通信的时候可以让另一mb去做计算。我们给每个物理rank都绑定了一个executor和一个active queue，为了达成这种目的，我们必须保证两个虚拟stage的active在**指令执行顺序没有依赖**和**申请的资源不会死锁**，这样才能让virtual stage A的通信和virtual stage B的计算相互overlap。

那么上面说的这种overlap要怎么排布active才能实现呢？这个必须先定义边界。

## 从dual（奇美拉）思考stage的执行序
dual V的思想是让一个物理rank不会只绑定模型的一层，而是模型的某两层。这个和vpp的区别是，vpp的优化是不同mb不同时做通信and计算带来的掩盖效果吗，而dual的思想是一个rank可以做几个model layer进而降低了单任务的IDLE，pipeline从两个方向同时发任务，气泡少。可能带来的代价是任务切换带来的开销，但直觉上这部分开销不会很大。（加个TODO先，有空也会验一验）

  TODO: 验证 folded V / VPP 下更细粒度 action interleave 带来的 kernel launch、stream sync、activation memory 和 comm fragmentation 开销。

## mb编码逻辑
随着foldv，dw分离，vpp的引入，mb的编码应该用一个结构体保存所有metadata而不是简单的id。

## action之痛
做到vpp的时候，开始发现femtotron这套基于active-mb的扩展性不够好，在做dw分离的时候我把一个backward action拆成一个d action和一个w action，其实这个设计就给后面的复杂埋下了问题。或者说一开始femtotron的forward，backward，sfrb等设计就让这种设计注定变得不可扩展。因为如果一个schedule包含多种优化，一个action的含义必然是复杂的。

所以没办法，先研究一下megatron是怎么实现的。






