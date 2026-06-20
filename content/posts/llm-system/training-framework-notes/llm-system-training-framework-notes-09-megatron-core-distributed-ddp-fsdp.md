---
date: '2026-06-20T00:00:00+08:00'
draft: false
title: 'LLM System: 训练框架随笔 09 - Megatron Core Distributed DDP 和 FSDP'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Training Framework", "Megatron", "MCore", "Distributed", "DDP", "FSDP"]
series: ["LLM System", "Training Framework Notes"]
series_order: 9
weight: 1
math: true
---

## 概述
core/dist接入了三种数据并行：torch fsdp2、mcore ddp、mcore fsdp

## FSDP

### 流程模型
FSDP就是一种sharding，sharding的对象是param、optim、grad

![FSDP 参数展平、补齐与分片后的 layout](../../images/llm-system-training-framework-notes/fsdp-parameter-layout.svg)

如图，fsdp的大概做法就是为了不让f/b期间大量模型的layer持续驻留在gpumem，所以会把这一个layer切到不同rank上，需要做f/b的时候再通过集合通信拿到。

这句话说完其实有两个质疑的点。一、如果我们把一次coll操作得到的整体作为一个unit，那么这个unit一定是上文中说的layer吗，可不可以是别的？可不可以是两层layer？二、其他rank如果也要做fsdp的shard，一样要把自己的全量参数/optim/grad切片并分发到其他rank，这么看只有对于一个rank的一个layer的fb计算，这一个小阶段内是可以用通信换存储。但是如果全局来看似乎并不一定省显存、因为本rank也负担了其他rank的shard？

一、 可以是别的但是一般默认一个unit就是一个layer
二、 肯定省显存，因为是一个dpgroup共享一个param/optim/grad副本。具体怎么共享看zero设计。

关于集合通信：ag一般是收集，rs一般是同步。
### Init
weight/param在shard之前有一个自己的layout，shard之后也有layout。这个layout非常简单，就是把所有参数拼成1D tensor。
![FSDP 参数、梯度与优化器状态分片流程](../../images/llm-system-training-framework-notes/fsdp-process.svg)
做成1D tensor的原因是为了适配dist.all_gather_into_tensor(),这个allgather接口生成的是一整个连续的shard，语义表达比较差但是性能好。而如果使用经典的dist.allgather会返回一个tensorlist，语义可读性更好，但是性能差，性能差的原因是他支持每个collrank传递不均等shape的tensor，但是fsdp的unit shard是均等的，因为没有额外的设置为不均等的理由。

reducescatter和reducescattertensor也是同理。

这个1Dtensor有自己的构建规则，mcore这里面是函数式构建，也就是通过设计特殊的递归，一类weight只能演算出一种1D layout。这种的好处就是不用手动指定metadata了，而且也不需要给每一种情况都写一种layout。

一个layer上很多weight，比如q k v norm这些东西，每个单独shard会有很多小包，所以用FlatParameter把这些凑成一个shard。用这个类去给刚才的1D tensor包一层。

### Runtime

流程为：preforward-forward-postforward-prebackward-backward-postbackward

![FSDP forward 与 backward hook 生命周期](../../images/llm-system-training-framework-notes/fsdp-hook-lifecycle.svg)

可以看到这里会注册两次hook，prebackward hook的操作是计算backward的时候需要一次ag把shard汇聚恢复。postbackwardhook的时候是需要一次rs同步梯度+释放shard。前向也会有类似的hook，区别是前向是主动写的hook，而反向的hook是前向调用时候才reg的。

为什么有这种区别呢。因为torch的设计是前向手动开始，反向由autograd按计算图执行，而torch是不感知到自己什么时候进入了某个layer的backward，不存在这种边界的定义，反向的时候torch只知道梯度来了要算梯度，所以得手动hook，原理上是如此，torch代码还需要再看看。

Runtime的时候峰值显存等于allgather之后拼好的unit+本地驻留的不被fsdp wrap跟踪的参数+平时就持有的shardSize（这个不能和全量unit算在一起 因为做merge实现太复杂了）

### lora策略
如果用peft或者lora，fsdp需要修改sharding策略，因为大部分权重是frozen的。

### fsdp通信计算重叠
fsdp带来的通信计算重叠从pp视角看是发生在stage内部。而ppstage间微弱的通信不是fsdp管理的。
fsdp的通信计算重叠来自于一个stage内部。当前layer的通信/计算和其他layer的通信/计算重叠。

![FSDP multi-stream 通信计算重叠示意图](../../images/llm-system-training-framework-notes/fsdp-multistream-overlap.svg)

看上面图就可以了，因为ag和compute unit是切片的所以能叠一部分。另外就是还有两类可选优化，f prefetch和b prefetch。

## ZeRO

![ZeRO-1、ZeRO-2、ZeRO-3 与 FSDP 核心区别](../../images/llm-system-training-framework-notes/zero123-vs-fsdp.svg)

## DDP
非常朴素，每张卡都有全量模型，每张卡处理一个mb。
