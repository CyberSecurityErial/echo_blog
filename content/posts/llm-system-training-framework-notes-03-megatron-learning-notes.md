---
date: '2026-06-16T00:00:00+08:00'
draft: false
title: 'LLM System: 训练框架随笔 03 - 再读 Megatron Core 看设计模式'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Training Framework", "Megatron"]
series: ["LLM System", "Training Framework Notes"]
series_order: 3
weight: 1
math: true
---

## 为什么突然就读上mcore了
从头给Femtotron加dualpipe的时候被自己的软件工程能力急哭，深刻感觉到想要0基础快乐vibe coding绝非易事，而且尤其是在其他人的项目上做增量开发的时候，一不小心就容易过度设计/破坏原有设计模式/重复设计。

所以就从pp schedule开始读。

## scheduler
其实scheduler在另一篇文章里面也说过，主要的优化就是这几个点：1f1b+vpp+dual+zerobubble。（moe的通信计算掩盖暂且不计入）
我在开发的时候想把这几个优化点的接口设计成某种“能力”，也就是把这几个优化点都做成下面这种模式：

default_scheduler = default_gpipe()
scheduler = zero_bubble(dual(vpp(1f1b(default_scheduler))))

这样做在接口上看起来很有美感也很理想化，但是我犯了一个很严重的错误。因为在规划阶段我的脑子里想的是可以像编译器排layout那样生成schedule pipeline。如果是对pipeline的整体做这个变换，那相当容易，但是问题就出在Femtotron的执行逻辑是SPMD的（Megatron也是这样）。SPMD具体是如何设计的，后面还会细说。这就导致了我们其实无法维护一个全局的controler，也就很难拿到全局的pipeline相对位置信息（可以拿，但是为此必须破坏大量的封装，保存大量的笨重结构体）。也就是说，这样设计scheduler在逻辑来看非常美，但是在实现上因为SPMD的设计，每个函数执行的范围是每个rank，输入的资源是一个chunk的model，因此保存全局的信息（对每个子块来说就是知道自己在全局处于什么位置）在实现起来非常繁琐，遂作罢。

于是本人暂时搁置了Femtotron的开发，开始参考Megatron-Core的代码实现。pipeline parallel的scheduler个人觉得是训练框架中比较有趣的一个内容，刚好也是在开发这部分，就借此机会把Megatron的核心源码走读一遍。当然可能写的比较草率，后续肯定还会单独整理一个比较完整的博客专门读。这里主要展示读的过程中我脑子里面的cot。

## pipeline parallel的一切
gpt5.5虽然被agenticRL搞得不说人话，但是有一个词用得很好导致我自己也很喜欢，“心智模型”。读各类代码尤其是涉及到并行计算的代码，都需要有这样一个心智模型。比如cuda是simt。编程模型这说法在框架这层就显得有点不够贴切。Megatron的pp代码，心智模型是SPMD，也就是每个Rank进入同一个函数。这个从megaton的使用方式里面也能看出来，启动方式类似于mpi。pp stage里面具体做什么操作，由rank固有的状态决定：包括这个rank属于哪个通信组。Megatron实现vpp的方式就和我预设的完全不一样，它在get_forward_backward_func（）里面先用ifelse判断要不要用1f1b，还是直接简单粗暴的gpipe，在1f1b的ifelse里面又直接用了ifelse判断要不要用vpp，而且给是否开vpp分别写了一个函数，这两个函数分别是forward_backward_pipelining_with_interleaving 和 forward_backward_pipelining_without_interleaving。

Megatron这样实现就可以体现出增量的困难，比如这里就无法设计成“把gpipe改成1f1b”或者把“1f1b增量成interleave 1f1b”的样子。也侧面说明我一开始的设想不适合整体代码架构。

### 执行层级

传统的深度学习一次train step需要准备一个batchsize的数据。而megatron里面则不会一次准备这样一个batch，而是每次只取出一个microbatch将其转化为gpu tensor然后做fb计算。cpu侧的dataloader可以做预取重叠，所以可以看到后面每次fbpp函数loop的时候都会基于data_iter拿到本次loop所需的mb。而不是整个batch。
总结就是fbpp函数在每个ppstage的每个trainstep都会调用一次。

### 无interleave实现： forward_backward_pipelining_without_interleaving
从上到下依次执行：
* 判断是否切分modelchunk，非interleave不支持切分chunk
* 判断是否是multi-module流水线，会使用过特殊的Communicator，（这个主要是为了区分是否为llm，因为llm Encoder-only不能做cp loss scaling）
* 判断是否启动了comm p2p overlap，非interleave不支持comm p2p overlap
* 初始化p2p_communicator（通信用）和pg_collection（保存各种p的group）
* 清空用于做dw分离的缓存。（上一次的清空，留着给这次用）
* 清空moe paged stash。（moe backward需要额外buffer）
* 禁用梯度同步disable_grad_sync，各个gpu计算梯度，但是不马上广播到其他gpu，这个是dp的东西但是要放在pp控制。
* 计算各个阶段microbatch的数量（warmup，steady，end）
* 根据是否为多模态模型选择backward的函数类型
* 根据mb，group，seqlen，decoderseqlen计算send和recv的tensor shape。decoderlen和seqlen的区别是有的模型是e-d而不是donly，所以需要特殊处理e-d结构的，donly的decoderseqlen就等于seqlen。
* 如果send和recv tensor形状不同，需要调用外部函数adjust_tensor_shapes_fn，这个具体函数目前只有一种实例化就是在做模型蒸馏的时候要传入一个get_tensor_shapes_adjust_fn_for_distillation函数，因为蒸馏的时候可能同时有teacher和student的tensor。需要特殊处理形状。（TODO详细了解）
* warmup阶段1：checkpoint_activations_microbatch 是否启用recompute（不用存前向acvitation省显存，属于是给1f1b打补丁）
* warmup阶段2: recvforward
* warmup阶段3: forward，如果是最后一个就算loss
* warmup阶段4: sendforward
* warmup阶段5： 检查如果是最后一个pp，那就计算本次mb的累计token，然后做几次广播获得全局token数。知道token数才能算梯度。有的rank token无法对齐，所以还是得广播才能知道全局有效token。
* warmup阶段6：保存本轮input和outputtensor到本地，deallocated（outputtensor）。这里是python伪释放机制，一个tensor有好几个成员，反向的时候只需要torch.autograd这个成员，不需要原始数据，所以deallocated可以实现只去释放.data而保留autograd，细粒度控制显存。如果调用torch绕不开，但是megatron直接用了c++ autograd engine。
* warmup阶段是rf+f+sf，那么还要r一次，才能正式进入1f1b的sbrf阶段。
* steady阶段1: 同warmup的1，重计算配置（TODO详细了解）
* steady阶段2: forward
* steady阶段3: sfrb返回grad（如果是纯forward就是sf）+ 存数 + deallocated，注意这里rb不代表真的收到了数据，是在等下游把梯度发回来。
* steady阶段4: 取出最早完成f的mb做b，这个得到的梯度是本rank的不是下游的。和上一条区分。
* steady阶段5: 如果是最后一个b，打开梯度累积。enable_grad_sync
* steady阶段6: backward
* steady阶段7: sbrf（sb）
* colldown阶段1：打开梯度累积。算完梯度会自动广播并reduce。
* colldown阶段2：弹出最近一个没完成的mb任务
* colldown阶段3：rb
* colldown阶段4：backward
* colldown阶段5：sb
* colldown阶段6：gradsync（注意这里看起来有很多gradsync的位置，但都是分支判断，实际只打开一次，而且必须保证是最后一个backward执行之前打开。如果在最后一个b之后打开，可能ddp和fsdp的backward hook错过了同步机会。）
所以实际上，就出现了代码里面的几种边界分支，如果没有cooldown backward的rank，比如说pp最后一次的stage，最后一次backward在1f1b steay之前就打开了，所以要在steady的最后一轮打开。而如果是有cooldown的rank，最后一次b是在cooldown的最后一轮。（TODO 画个图更清晰～）
* 把dw分离里面没计算的w都计算了。注意整个流程里面并没有算d存w的过程，因为这个过程隐藏在了forward和backward的实现里面。
* dp梯度同步+ppsp梯度同步+globaltoken缩放梯度。

### 有 interleave / VPP 实现：forward_backward_pipelining_with_interleaving
基本实现都差不多，依次有几个点不一样

* 如果k个chunk和m个mb，一共要过km次。这个地方会生成一个调度表去显式写好每一次哪个mb和哪个chunk匹配。

### moe paged stash
众所周知forward完以后有一些metadata不能丢，等backward的时候还得用。这个metadata的结构在torch层做了定义，叫做autograd。megatron就负责管理这部分metadata放在哪。torch作为一个求导引擎，具体怎么用metadata的这里先不管。但是在megatron层面不能丢掉autograd就意味着显存一直占用，训练最需要注意的就是显存峰值，毕竟要注意oom。如果f和b距离太远那显存就一直占着，很危险。
autograd存在的问题是分配了大量冗余的memory。autograd的粒度是算子，也就近似是一个ffn，autograd记录并分配的tensor形状是capacity✖️hidden，（那为什么autograd设计的这么蠢）

那么就自然而然地想问，为什么一开始设计的这么蠢。原因有几个，alltoall需要提前分配buffer所以只能用capacity，做groupgemm就需要shape一样，动态shape需要写大量kernel，动态shape用不了cuda graph。

pool确实没办法把这个capacity的限制压下去，但是可以用池化的方法。分配的物理buffer还是那么大，但是可以通过分页。物理buffer是malloc出来的，然后需要知道一个概念叫做“当前可用的显存”，当前可用的显存=malloc出来的-已经被占用（锁住）的显存。如果不分页，就相当于已经被占的显存大了很多。所以当前可用的显存会变少。而实际上那些被其他tensor占的显存块里面是有冗余的。
### overlap p2p comm
放下一期讲 单独说通信器
### batch p2p comm
同上
### overlap_p2p_comm_warmup_flush
同上
### pp里面如何处理通信hang住问题（专题，nccl+mgt）
同上
### megaton pp适配多模态
主要是encoder only，decoder only和encoder-decoder在seqlen和sp的区别。todo。
### megatron pp适配蒸馏场景
考虑teacher和student。todo。
### dw分离的w插在哪里？
为了掩盖pipelineflush的开销会把wgrad单独计算。所以要在colldown做的过程中也去计算dw。然后所有cooldown的mb都做完了之后调用一次dwfinish。