---
date: '2026-06-19T00:00:00+08:00'
draft: false
title: 'LLM System: 训练框架随笔 07 - MCore Export'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Training Framework", "Megatron", "MCore", "Export"]
series: ["LLM System", "Training Framework Notes"]
series_order: 7
weight: 1
math: true
---

## export概览
export模块做的事情就是需要给用mcore训练出来的模型做一个推理优化，推理框架是trtllm。（为啥，啥地方用？）
那么就得考虑几个问题：
1. 为啥要推理优化，和训练过程中的forward啥区别，什么情况会用。
2. 这里面既然是做推理优化，那么有哪些优化是和训练场景耦合的？哪些优化是必须妥协不能做的？哪些优化手段可以被做成任意推理场景都能随时插拔的模块？
3. 那么既然是在看训练框架，我们最应该关注的就是和训练相关的推理优化了。那么和训练相关又有几种情况呢？我认为是两种。首先是训练框架本身对模型ckpt的设计会影响推理框架如何去解析这个模型，比如说我们可以猜想一下，如果ckpt是分布式的，那么trtllm大概率也会用一种分布式的方式去读写ckpt，或者说如果ckpt在内存layout上并非推理框架最友好的，那么还会产生一次转换等。其次是训练相关的一些上游任务，还是一开始的问题，为什么训练框架里面要内嵌推理框架，这部分推理加速会被用在什么地方，这个推理的地方会有什么样的workload特点？

## 总结
gqa和mqa的推理tp切法必须要用ckpt里面gqa和mqa的设置来限制，不能随便改Q/K。

layout的重排全都是trtllm离线自己做，会把训练ckpt的默认layout转换成推理友好的形式。

训练做完量化之后推理不用再重新calibration（校准范围），只需要复用之前的factor就行。

训练的tp/pp shard和推理的tp/pp shard不一样，也相当于一种分布式的layout（？可能这么理解不是很严谨）。

dist-ondevice convertion，转换成推理权重的时候需要考虑ckpt已经是多卡shard了，如果shard已经做了分布式，那么对它的操作最好也是分布式的，否则会造成很大的barrier。这个道理和mcore dataset里面计算每rank的data idx类似，只不过那个却是让一个rank算全局，其他rank要等着。但那个东西设计成barrier有其他的原因（分布式写共享文件容易导致更严重的抢占，所以还是一个rank做single-writer，这里还带了一个很重要的常识，你如果想多线程一起算一个东西，必须至少还得维护一个共享的缓存来同步他们的计算结果。当然也可以每个rank都算一遍全量，那就是用不到共享存储了），当然既然牺牲了分布式操作肯定也带来这里所说的问题。

既然训推的tp不保证一样，tp又对padding又要求，按vocabsize这一列去切的时候必须能整除tp，所以训推的vocabsize的padding就不一样。所以还得有一步转换问题 相当于训推reshard的副产物。
