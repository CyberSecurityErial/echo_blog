---
date: '2026-06-24T00:00:00+08:00'
draft: true
title: '工程踩坑：用 Ray 分布式启动 NCCL 的巨额冷启动开销问题'
categories: ["工程踩坑"]
tags: ["Ray", "NCCL", "分布式训练"]
series: ["工程踩坑"]
series_order:
---

单机八卡，Ray启动训练任务发现启动很慢，具体慢在torchdist的barrier()，然后做了几组对比实验。分别是经典torch.run，torch.run(virtual_visible)，Ray。
发现torchrun是正常速度，但是后两者firstbarrier有几秒的时间。
Ray为了资源隔离，设计的进程模型是只能看到一张卡的形态。认为是这个设定导致了overhead。
torchdist的barrier由于在整个训练流程的入口，也会承担建立通信域（NCCL视角是Comm，Torch视角是processGroup）的功能，直观来看就有可能是初始化通信域太满了，这是一个猜想。

深入到torchdist的代码发现barrier做的事情是dist.all_reduce()，这个里面又做了initProcessGroupNCCL-initNCCLComm-initTransport-enqueueNCCLKernel-AsyncWaitKernelQueue
其中从initNCCLComm开始都是NCCL的内部代码。

但是这个还是不好拆，因为dist.all_reduce是一个AsyncCallback，所以得进一步再往下拆，是丢任务的过程慢了还是任务本身执行慢了。所以这样测：
  work = dist.all_reduce(tensor, group=group, async_op=True)
  work.wait()
返回work之前的时间是丢任务这个过程链路的时间（也就是enqueue以及之前），wait的时间是collkernel执行时间。最后发现是返回work之前很慢。

所以深入NCCL建立Comm的过程，问题出在cudaDeviceGetPCIBusId这个函数。

首先需要了解本文语境中的设备可见性问题。这里的设备可见性分为几层：
1. CUDA runtime可见性
这个是进程级别的，由CUDA_VISIBLE_DEVICES这个环境变量决定，同时也决定了runtime接口能不能看到足够多的GPU。
2. NCCL peer可见性
这个是逻辑级别的，代表的是当前rank需要通信的另一个GPU，不考虑物理上是否可达，只描述NCCL要和谁进行通信。
3. NCCL Topology / NVML可见性
这个是设备物理连接级别的，需要收集busId，nvmlDev信息。

现在NCCL建立Comm的时候用的是runtime接口，而如果我们设置了GPU资源的隔离，runtime这一层就真的感知不到busId，而不会选择去找物理拓扑。NCCL对于peer逻辑上可见但找不到物理busId的情况，设置了一种分支状态叫做invisible P2P，在cuda 10.1之后支持。Ray就是触发了这条比较少触发的路径。

那么为什么触发了这个路径会变慢呢？因为为了p2p又不能基于runtime接口直接hack到busid，就要做一些跨进程共享机制，也就是NCCL语境下的p2p。

看p2p.cc可以发现跨进程通信有两类，一类是cuMem，一类是IpcHandle。默认走的是cuMem。二者区别在于cuMem是一种eager模式，而IpcHandle是一种lazy模式，所以cuMem一开始慢，但是后面会更快。解决办法就是关掉cuMem。因为目前场景没有看到显著的IpcHandle拖累的现状，后续有需求再改。