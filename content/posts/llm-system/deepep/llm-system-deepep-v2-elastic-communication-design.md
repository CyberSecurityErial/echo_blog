---
date: '2026-07-28T17:08:27+08:00'
draft: false
title: 'LLM System: DeepEP v2 弹性通信架构图解'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "DeepEP", "MoE", "NCCL", "CUDA", "RDMA", "NVLink"]
series: ["LLM System", "DeepEP"]
series_order: 1
weight: 1
math: false
---

> 这篇笔记把 DeepEP v2 的对象、内存、同步和数据路径压缩成五张结构图。正文只补充图中不适合展开的语义边界。

## 架构总览

![DeepEP v2 弹性通信架构总览](/images/llm-system-deepep-v2-elastic-communication/01-architecture-overview.png)

`ElasticBuffer` 管理 Dispatch / Combine 所需的通信与 Buffer 资源；`EPHandle` 保存一次 EP 数据传输的元数据，包括确定性模式、Dispatch 缓存和 GroupGEMM 的 Token Padding 信息。

## ElasticBuffer：物理分段，虚拟连续

![ElasticBuffer 混合内存与连续虚拟地址](/images/llm-system-deepep-v2-elastic-communication/02-elastic-buffer-memory.png)

CPU 与 GPU 内存物理上分段，但通过 CUDA VMM 映射为连续虚拟地址，再注册为 NCCL Window。这里的 Window 提供跨 Rank 访问语义；它本身不等同于 NVSHMEM 原生的对称内存抽象。

## Barrier 与通信计算重叠

![Barrier 跨 Stream 与跨 Rank 的两级同步](/images/llm-system-deepep-v2-elastic-communication/03-barrier-overlap.png)

图中展示的是全局 Barrier 的基本心智模型。Hybrid 分层 Barrier 的并行模式还存在一个容易误判的弱同步边界：A0 分别看到 B0 的 Scale-out 到达和 A1 的 Scale-up 到达，并不能推出 B1 已经到达。局部可见的两条同步事实不能自动合成为全局强 Barrier。

`async_with_compute_stream` 允许 Barrier 延迟返回 Event，在真正消费通信结果之前继续执行独立计算；`prefer_overlap_with_compute` 则通过减少通信占用的 SM，为计算留下资源。

## Dispatch / Combine 数据路径

![Elastic CUDA Dispatch 与 Combine 数据路径](/images/llm-system-deepep-v2-elastic-communication/04-dispatch-combine-data-path.png)

Direct 是单跳传输，不需要构造两跳的 Warp 生产消费流水；Hybrid 则由 Notify Warp、Scale-out Warp 和 Forward Warp 分别承担控制、RDMA 生产与 NVLink 消费。Combine 最终依赖 Metadata 恢复 Token-major 布局并完成 Reduce。

## 从 Legacy 到 Elastic

![Legacy 到 Elastic 的统一接口与调优空间](/images/llm-system-deepep-v2-elastic-communication/05-legacy-elastic-tuning.png)

Elastic 的重点不是消除 LL 与 HT 的 Workload 差异，而是先统一 Buffer 和 Dispatch / Combine 接口，再让 AutoTune 与 JIT 根据场景确定 Kernel、Buffer、SM、QP 和 Channel 配置。统一抽象换来了更大的 Buffer、更多 QP 状态，以及 EP Kernel 和 `EPHandle` 的额外资源占用。
