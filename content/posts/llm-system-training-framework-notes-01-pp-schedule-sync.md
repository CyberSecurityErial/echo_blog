---
date: '2026-06-10T00:00:00+08:00'
draft: true
title: 'LLM System: 训练框架随笔 01 - PP Schedule 为什么要做成非异步的'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Training Framework", "Pipeline Parallel", "Schedule"]
series: ["LLM System", "Training Framework Notes"]
series_order: 1
weight: 1
math: true
---

> 本篇目标：

# schedule要不要做成异步
## gpipe的非异步调度
就是简单的做sf和rf，sf和rf之间依赖于torch.dist的api做阻塞同步。
在其他博客中也说这种静态调度和下发cpu指令差不多，按顺序一条条执行，执行完了整个程序就跑完了。
## 异步的问题
那为什么不直接把任务丢到下游（backward是上游）然后直接做下一个mb的计算呢，这样看起来还可以让sm利用率高。
搞成异步之后快的那个stage确实会更快推进，但是慢的那边会更慢。要么就是两遍差不多快，一样没有什么提升。
另外就是action memory的值会比较不确定。可能会非常大。而静态调度actionmemory的大小是可控的。
## 语义问题
如果搞成异步下发，那说明会有任务的积压，这些任务做了一半以后，checkpoint要按照哪个标准做记录呢？这也是很难做的。
静态调度的状态就是比动态调度状态更少。动态调度存储，加载，更新状态都会更难而且可能bound在控制流。
## 好处？
动态调度对慢节点的容忍度好，但是绝对不是pp schedule pipe里面的慢节点。因为llm场景每个mb的时间都差不多。就算是卡慢了也不会这样处理，直接换掉就行了。这种对慢节点的容忍度指的是对于一些异构的流程，比如说rl的几个步骤，以及搜推处理sparse数据等。但是这两个流程我都暂时不特别熟悉，后续还要继续学习。

# 如果做同步pipe，算子做还是框架做？
## 放在框架
那就是调用action之前barrier一下，因为这样涉及到多节点，所以启动开销会比较大，好处是位置浅好定位。
## 放在算子
那训练框架侧就只启动torch.dist的接口，torch.dist底下再接入通信算子库。算子内部barrer。这种问题是调用栈会很深，以及算子级更难定位，好处是算子层可以做更深度的优化，比如做smfree把单节点的mfu打上去。（不过还是那个问题，单节点mfu可能真高了，全局不好说）
那其实这里再给自己开个新todo，试试sm free的算子实现，用CE做通信，到时候跑训练看下效果。
