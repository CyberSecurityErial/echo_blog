---
date: '2026-06-18T00:00:00+08:00'
draft: false
title: 'LLM System: 训练框架随笔 05 - Megatron Checkpoint'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Training Framework", "Megatron", "Checkpoint"]
series: ["LLM System", "Training Framework Notes"]
series_order: 5
weight: 1
math: true
---

> 本篇目标：

## megatron Checkpoint

checkpoint包含如下四类：

rng state/rerun state/dalaloader state/model&optim state

rngstate是一些伪随机数的序列，因为伪随机数的采样本身就是很好的分布，但如果ckpt没有记录他们的状态，会破坏这种分布，进而给训练带来一些不可预估的问题。

rerun state是用于做容错的。

dataloader state是数据的ckpt，表示训到哪里。

model/optim state是模型权重，优化器状态，学习率衰减程度等算法层直接可见的东西。

## RNGstate详解

```python
rng_state = {
      "random_rng_state": random.getstate(),
      "np_rng_state": np.random.get_state(),
      "torch_rng_state": torch.get_rng_state(),
      "cuda_rng_state": torch.cuda.get_rng_state(),
      "rng_tracker_states": get_cuda_rng_tracker().get_states(),
  }
```

random_rng_state： python的random模块 如random.random() .shuffle() .sample() .randint()

np_rng_state：   np的random。如np.random.random() np.random.shffle() np.random.randint()

torch_rng_state：torch的random，如torch.rand(), torch.randn(), torch.randint() 前提是device=cpu

cuda_rng_state： torch.rand(cuda) torch.randn(cuda) F.dropout(cuda_tensor) cuda_tensor.normal()

rng_tracker_states： gpu cuda有随机数的流，默认情况下所有gpu的random公用这一个流，但是这个不够好。所以会以类似于上下文的方式，去管理随机流。这个state记录的就是各个随机数流推进到了哪个位置。

### 如何发现rngstate出问题了？

既然框架设计了这个功能，就要思考这个功能如果没做或者没做好，带来什么问题。如果是rngstate没同步好，那么是否启动save/load ckpt就是最先需要判断的标准。所以一开始可以先单卡+是否saveload。

## rerunstate详解

## megatron checkpoint backend - mcore dist checkpoint

ckpt前端其实只是决定要存哪些状态。后端决定了怎么在多rank存储。

ckpt的后端是distckpt，distckpt这一层看到的是一个全局的共享文件。distckpt只需要提供每个rank的shard在全局的位置，以及告诉本rank其他rank的一些shard信息。本rank就可以读取各种shard，这一层是屏蔽掉各种近端远端读写逻辑的。

训练框架这一层就只看见一个共享目录，调用的话就是系统调用，非常简单。其他的底层细节让xxFS去处理。

具体怎么shard由训练参数决定策略。策略指定的是谁保存shard，谁读取shard，是否需要在rank之间重分配io（比如由其他rank代读取），同步还是异步write让，是否缓存metadata和plan。

plan是本次ckpt的任务靠舒服。同步写入的话需要让训练和写入阻塞，异步写入就是后台去做。metadata就是distckpt提供的索引，每个rank可以根据这个去拿。这个缓存策略指的是把每次plan和metadata都放在rank对应的cpu进程的内存里面，如果plan不变，就可以跳过创建plan，收集其他rankplan，省掉一些metadata的同步和plann的创建latency。

那么什么时候plan会变化呢，主要就是sharding改变的时候需要变。就不能跟据这个做cache了。
