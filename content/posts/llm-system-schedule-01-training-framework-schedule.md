---
date: '2026-06-08T00:00:00+08:00'
draft: false
title: 'LLM System: Training Schedule 01 - 训练框架中的 Schedule 算法'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Schedule", "Pipeline Parallel", "Distributed Training"]
series: ["LLM System", "Training Schedule"]
series_order: 1
weight: 1
math: true
---

> 本篇目标：

## 问题背景
什么是schedule，这个词含义很广但是在训练框架这里一般考虑的是f和b任务之间的编排。

## PP
pp开几一般就是把所有layer除以几，然后每个就是一个stage的layer数量。一般按照layer切。
## GPipe
有很多mb，每个mb要做很多stage（模型的layer或op，跨卡或跨机，这些都行），GPipe就是要等到所有的mb都做完他们自己的所有前向stage，然后开始反向stage。气泡比较多，此外因为前向和反向的layer是反过来的，所以对于一个mb来说，他做的这些stage里面做前向越早的那个stage，做反向越晚。也就是inflight越多。inflight越多就代表得保存中间的状态，占显存。所以2个肉眼可见的缺点一个是空泡另一个是inflight。

[在 Perfetto 中打开 GPipe trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/gpipe_trace.json)

## 1F1B
做F的预取，然后让F和B同时进行。中间的卡交替进行f和b。好处是inflight少，但是空泡不减。

[在 Perfetto 中打开 1F1B trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/1f1b_trace.json)

## Chimera
最接近dualpipe的办法。
初始流水线：s0f-s1f-s2f-s3f-s3b-s2b-s1b-s0b。
Chimera主要减少了bubble，前面二者有bubble都是因为GPU来任务的时间难免有pipeline式的三角形空泡问题。但是三角形空泡来源一个先入为主的假设就是我们总假设只能gpu0开始做mb0stage0。如果让其他gpu也同时开始一个任务，三角形空洞就能补上很多。（拓展，Chimera只是同时走两段pipe，能不能更多的pipe，收益如何）

[在 Perfetto 中打开 Chimera trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/chimera_trace.json)

## zero-bubble
dualpipe也利用了这里思想，一句话就是做b阶段的bw分离。
前向：

$$
Y = XW
$$

反向：

$$
dX = dY W^T,\quad dW = X^T dY
$$

（dY直接当已知）
dX是他前序的dY，所以真有依赖的就是dX，但是传统方法都是dWdX一堆算出来，这样就导致粒度不够细。dX是关键路径上的依赖，但是dW不是，所以把dW拆走放在有bubble的时候去填充bubble。这里代码实现需要思考下怎么填。这里好处就是1.可以把F和B的运算强度给打平。只算一次gemm（当然实际上肯定不可能精确的打平，这里不考虑，因为考虑了也没办法更进一步优化了）2. 可以把尾部的三角形pipeline填上w。

[在 Perfetto 中打开 ZeroBubble 1F1B trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/zerobubble_1f1b_trace.json)

## interleaved 1F1B
也叫vpp，把一个stage再划分为几个虚拟stage，用interleave的形式排到几张卡上。这个场景为什么能减少bubble在我第一次理解的时候其实不是很直观，因为我思考的是，就算切细了那三角形的空泡依然存在，为什么空泡会少。所以就计算了一下size。只算开始部分的三角形空泡（结束时候是对称的就不管了）不计算的很细的话，我们看三角形空泡里面最长的部分，也就是最底下的那条，长度正比于每个f/b的时间*（pp-1），但这里要注意一个很容易想当然的问题，这里的pp是物理pp数，也就是真实的stage，而不是虚拟的stage。因为我们真实的stage数量一般和gpu数量一样，所以就算很多虚拟stage，一次填充到流水线的阶段也最多只有物理个gpu数。那切完以后f/b的t就变小了，显然空泡就小了。
这么解释不太直观，，最直观的其实是，让最底下的那个rank早启动。假设就是rank0到7，rank7得等好几个阶段才能启动，那就把阶段切细，然后启动的就快了。但是如果切得太细，跨rank（其实是stage）通信不能忽略，那就也不行。

然后写这个还想一个问题就是stage到底跨卡还是跨机还是跨什么东西，问了下ai说具体情况具体分析（等于没说）这个后续再看。
## DualPipe 
dualpipe首先吸收了Chimera和zerobubble的特点。做了双向流水线+bw分离。然后还对moe做了优化。

[在 Perfetto 中打开 DualPipe trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/dualpipe_trace.json)

## DualPipeV

[在 Perfetto 中打开 DualPipeV trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/dualpipev_trace.json)

## 为什么不考虑通信？

## 削峰填谷

## moe有何不同

[在 Perfetto 中打开 MoE bad overlap 1F1B trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/moe_bad_overlap_1f1b_trace.json)
