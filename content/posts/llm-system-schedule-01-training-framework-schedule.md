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

如何实现一个GPipe呢？（底层组件假设已经分好了，我们只需要考虑怎么把任务排好发出来，底层组件的事情可以见后文如何实现一个调度器）
非常的简单，给每个stage执行的载体（GPU）从mb0下发到mbn就可以了。然后执行那个stage对应的layer的前向传播/反向传播。

## 1F1B

做F的预取，然后让F和B同时进行。中间的卡交替进行f和b。好处是inflight少，但是空泡不减。

[在 Perfetto 中打开 1F1B trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/1f1b_trace.json)

如何实现一个1F1B呢？也并非很难，假设我们的stage执行载体（GPU，虽然总是括号里写GPU但是某些场景不一定是GPU，目前为了便于理解先这么写）
有m个，那只需要给stage编号（这个编号代表第一次启动任务的顺序）为i的stage提前分配m-i+1个mb就行了，mb的序号是从0到m-i。
然后这样预填充完之后，只需要做简单的配对+交叉即可。因为是1f1b，所以只需要交替下发f和b任务，f和b任务对应的mb编号只需要匹配最近一次任务即可，如果是b则找最旧的未完成mb任务id让fb闭合，如果是f则找最新的未完成mb任务id+1。

## interleaved 1F1B

也叫vpp，把一个stage再划分为几个虚拟stage，用interleave的形式排到几张卡上。这个场景为什么能减少bubble在我第一次理解的时候其实不是很直观，因为我思考的是，就算切细了那三角形的空泡依然存在，为什么空泡会少。所以就计算了一下size。只算开始部分的三角形空泡（结束时候是对称的就不管了）不计算的很细的话，我们看三角形空泡里面最长的部分，也就是最底下的那条，长度正比于每个f/b的时间*（pp-1），但这里要注意一个很容易想当然的问题，这里的pp是物理pp数，也就是真实的stage，而不是虚拟的stage。因为我们真实的stage数量一般和gpu数量一样，所以就算很多虚拟stage，一次填充到流水线的阶段也最多只有物理个gpu数。那切完以后f/b的t就变小了，显然空泡就小了。

这么解释不太直观，，最直观的其实是，让最底下的那个rank早启动。假设就是rank0到7，rank7得等好几个阶段才能启动，那就把阶段切细，然后启动的就快了。但是如果切得太细，跨rank（其实是stage）通信不能忽略，那就也不行。

然后写这个还想一个问题就是stage到底跨卡还是跨机还是跨什么东西，问了下ai说具体情况具体分析（等于没说）然后翻了下之前（未发布）的训练框架学习笔记，原则上stage没有跨什么东西的限制，但是在机内有高速互联的情况下一般是跨节点的。因为高速互联要留给tp。优先级tp>dp>pp因为我们假设tp每一层都开一次，那么tp的通信量是

$$
seq\_len \times batch\_size \times layer \times hidden
$$

dp没有layer这个维度肯定要少点。pp一般都可以overlap了。如果节点内没有高速互联是需要开pp的。

## Chimera

最接近dualpipe的办法。

初始流水线：s0f-s1f-s2f-s3f-s3b-s2b-s1b-s0b。

Chimera主要减少了bubble，前面二者有bubble都是因为GPU来任务的时间难免有pipeline式的三角形空泡问题。但是三角形空泡来源一个先入为主的假设就是我们总假设只能gpu0开始做mb0stage0。如果让其他gpu也同时开始一个任务，三角形空洞就能补上很多。（拓展，Chimera只是同时走两段pipe，能不能更多的pipe，收益如何）

其实就是排两个交叉的流水线。依然要vpp把stage加倍。

假设原先4stage，vpp成8个。

```text
s0 0 7
s1 1 6
s2 2 5
s3 3 4
```

就大概这样的。

如果纯做vpp的话是这样的：

```text
s0 0 4
s1 1 5
s2 2 6
s3 3 7
```

那为什么vpp的效果不如Chimera呢。可以观察一件事情，stage越“在时间上靠前”被下发做f的，在做b收口的时候越晚，占用的显存就越大。所以有一个直观的结论，不同stage的显存开销在时间上是不均匀的。最影响显存开销的就是f的第一stage-b的最后一个stage这一对。这一对fb启动最早释放最晚，所以我们如果多同时启动几个这样的f，就可以让显存开销在时间上更均匀，进而降低了显存需求量的峰值。我们做vpp的话很难让不同stage的显存分配量是均匀的，甚至还有可能让inflight叠加。

[在 Perfetto 中打开 Chimera trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/chimera_trace.json)

下面思考actions如何排布，这就是一个纯粹的算法题问题了，m个stage，km个虚拟stage，办法就是stage/2对半做对称。然后f和b继续用堆栈做匹配就可以。

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

如何实现呢。这个要把backward拆一下，拆成bx和w。然后bx用于替代之前的b，做正常的1f1b逻辑就行。bw用来填充末尾的流水线三角形。

还有一点要注意就是一定要让bw在bx之后，否则还是阻挡流水线没意义。

## DualPipe 

dualpipe首先吸收了Chimera和zerobubble的特点。做了双向流水线+bw分离。然后还对moe做了优化。

[在 Perfetto 中打开 DualPipe trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/dualpipe_trace.json)

所以实现上其实就分为三块，双线+bw分离+moe特化。这一节就只谈moe特化。moe引入之后会出现什么问题呢？多了两次alltoall。时间开销大。

针对moe的特化主要是通算重叠+通算重叠的补丁。

如果对moe层做1f1b的话顺序是：attnF-dispatchF-mlpF-combineF-dispatchB-mlpB-combineB-attnB

这可以发现计算和通信可以同时做：

```text
计算队列    attnF mlpB mlpF attnB
通信队列    combB dispF dispB combF
```

为什么是这么排呢，相当于把：

```text
attnF   mlpF
    dispF   combF
和
combB   dispB
    mlpB    attnB
```

重叠在了一起。

但新的问题就是，F做完了combF才会发给下一个stage，这就是虽然内部overlap了，但是overlap完形成的一个stage上的一个任务条带无法和其他任务条带进行overlap了。所以会有气泡。

所以dualpipe里面把一个任务条带的通信计算overlap改成了：

```text
计算队列    mlpB_j mlpF_i attnB_j attnF_{i+1}
通信队列    dispF_i dispB_j combF_i combB_{j+1}
```

i和j是“配对”的，配对指的是两个能被融合在1f1b的mb。

按照这种实现，做完combF_i就可以往下一个stage发forward（isend）了。

然后可以发现的是，刚好alltoall通信很久，所以zerobubble的bw分离思想在这里更好用了。

```text
计算队列    mlpB_j mlpF_i (mlpBw_j) attnB_j (attnBw_j) attnF_{i+1}
通信队列    dispF_i dispB_j combF_i pp combB_{j+1}
```

insight: 一个moe layer里面计算顺序不能改，但是多个layer连着执行就会成为一种环（

## DualPipeV

[在 Perfetto 中打开 DualPipeV trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/dualpipev_trace.json)

就是加了vpp的dualpipe，如果保证虚拟stage恒定那就是用的物理stage载体数dpv要更少。有和vpp一样的问题，就是如果切得太碎会让点对点通信成为bubble。

## 为什么不考虑通信？

通信被掩盖？还是通信时间短？还是考虑了通信之后没意义？

## 削峰填谷

dual的奥秘，还有ringattention也是类似的，还有nccl的ring。

## moe有何不同

[在 Perfetto 中打开 MoE bad overlap 1F1B trace](https://ui.perfetto.dev/#!/?url=https://CyberSecurityErial.github.io/echo_blog/traces/moe_bad_overlap_1f1b_trace.json)

1. alltoall和计算kernel执行时间差太多
2. 反向传播的计算方法。

## 如何实现一个调度器

直接看各种论文里面的“贴瓷砖图”，横轴时间，纵轴Stage，方块上编号是一个mbid，这种图很显然说明一件事：任务是提前排好的。只看一个stage或者说只看一个GPU上面的条带，其实就是一大串简单的指令。也就是说调度器把指令发下来之后，只需要维护一个东西动态轮询上游信号，上游准备好了就去做，做完了发给下游。

上文提到如果逐个条带去观察rank的行为其实就是一串指令，但是还有一个非常不能忽视的东西是stage之间的依赖。这部分依赖怎么管理会比较好。

所谓依赖说具体点就是等上游数据，那我们两种做法，一种是让本地的stage执行载体（GPU）不感知全局的依赖，进了actions就一直等，直到数据来了。另一种是维护一个DAG执行依赖图，如果依赖不满足，就阻塞actions。其实本质区别是阻塞在actions内部还是actions前面的区别。actions再往底层就是通信算子，那这个或许就可以成为一个可以探索的问题。因为如果进通信算子再等待，可能会带来一些device的开销（取决于通信算子怎么实现的），而如果在actions之外就等待，维护DAG的一致性以及保存DAG的方式又是一个值得思考的问题。

在Femtotron中，使用的方案是在通信算子内部阻塞，也就是第一种。（其实训练框架这层如果用torch.dist那么就只会感知到torch.dist.isend这一层）
