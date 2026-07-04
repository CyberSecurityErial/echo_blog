---
date: '2026-07-04T18:23:55+08:00'
draft: false
title: '工程踩坑：torch.empty 和 torch.zeros 的性能区别'
categories: ["工程踩坑"]
tags: ["PyTorch", "性能"]
series: ["工程踩坑"]
series_order:
---

## torch.empty() torch.zeros()语义区别
区别很简单，是否初始化。

性能上，empty后的tensor分页操作是lazy的，lazy模式不总是好的。

## lazy（empty）严重劣化的case--RDMA（或其他需要pinned mem场景）
如果是empty，内核给torch操作所在进程分配page是lazy的。如果没有真的分配地址，相当于进程只持有虚拟地址。如果rdma恰好要reg这块地址。
rdma调用ibv-reg-mr会导致os先处理缺页，rdma缺页的处理比普通用户更慢，因为rdma一般要pin很大的buffer，普通写入时缺页一般只是一个page。另外rdma分配好page之后还要pinpage，做dma映射，最后才能建立mr。

**其实rdma场景烂的本质原因就是lazy设计带来的外挂问题太多了，rdma的缺页操作后面挂了一串东西：build pte--pin--dma mapping--create mr，lazy可以，但是如果lazy的东西太长，无法被overlap，就容易出问题，rdma正是有这样的问题。**

所以如果你的tensor要被rdma取用和发送，就一定要先用zeros初始化一下，保证真的有物理page。
