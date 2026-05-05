---
date: '2026-05-05T10:30:00+08:00'
draft: false
title: 'LLM System: PD 分离 00 - 学习地图'
categories: ["AI"]
tags: ["LLM", "LLM System", "Serving", "PD Disaggregation", "KV Cache"]
series: ["LLM System", "PD Disaggregation"]
series_order: 0
weight: 1
math: true
aliases:
  - /posts/llm-system-01-pd-disaggregation-map/
---

> 这篇文章是 LLM System 系列里 PD 分离子专题的第 0 篇，也是这个主题的学习入口。是笔者让gpt-5.5通过联网搜索帮自己制定的系统性学习方案。笔者会根据这个方案来确定如何学习PD分离的整套机制。目标不是先把所有论文细节读完，而是先建立一张可以持续填充的地图：该读什么、该推导什么、该写什么代码、最后应该能回答什么问题。

这个系列暂时围绕一个问题展开：**为什么现代 LLM serving 系统越来越关心 prefill/decode disaggregation，也就是 PD 分离？**

我希望自己最后能回答四个问题：

```text
1. 为什么 prefill 和 decode 要分离？
2. 一个 workload 到底该配多少 P worker、多少 D worker？
3. KV cache 从 P 到 D 传输到底传了什么、代价多大？
4. vLLM / SGLang / Mooncake 里这件事具体怎么落地？
```

先说一个结论：**PD 分离不是一个“拆进程就能变快”的魔法优化。**它真正解决的是服务系统里的资源解耦问题：prefill compute、decode iteration、KV cache 生命周期、网络传输和调度策略，本来在 colocated serving 里被绑在一起；PD 分离试图把它们拆开，让不同阶段按照不同目标优化。

## 0. 心智模型

LLM 推理一个请求大致分成两段。

**Prefill**：一次性吃掉 prompt，生成整段 prompt 的 KV cache，并产出第一个 token。长输入时它更像大 GEMM，通常更容易把 GPU 算力吃满。它最直接影响的是 **TTFT**，也就是 time to first token。

**Decode**：每次生成一个 token，每一步都要读历史 KV cache 和模型权重，并且自回归串行迭代。batch 小时它更像 memory-bound / latency-bound workload。它最直接影响的是 **ITL / TPOT**，也就是 inter-token latency / time per output token。

所以 PD 分离的直觉是：

```text
prefill 目标：尽快处理输入，降低 TTFT
decode 目标：稳定逐 token 生成，降低 ITL / TPOT
```

Splitwise 从硬件异构角度切入：prompt computation 更 compute-intensive，token generation 更 memory-intensive，所以可以把两个阶段放到不同机器或不同卡型上。DistServe 从 serving SLO 角度切入：prefill 和 decode colocate 时会相互干扰，并且 TTFT / TPOT 的资源和并行策略不一定相同。Mooncake 则把问题继续推进到 KVCache-centric 架构：不只是 P 节点算完、D 节点继续 decode，而是把 KV cache 当成整个 serving 系统的中心资源来管理。

但是这个优化有边界。vLLM 文档明确提醒，disaggregated prefilling 主要用于分别调 TTFT 和 ITL、控制 tail ITL，并且当前功能是 experimental；文档还直说它 **does not improve throughput**。因此我更愿意把 PD 分离理解成一种 SLO 和资源组织工具，而不是一个默认提高总吞吐的技巧。

## 1. 先读 serving 基础

不要一开始就冲 Mooncake。Mooncake 里的很多设计默认你已经理解了 LLM serving 的调度和 KV cache 管理。

第一篇应该读 [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)。Orca 不是 PD 分离论文，但它奠定了生成式模型 serving 的基础：**iteration-level scheduling**。传统 request-level batching 把一个 request 当成不可拆分的单位，而生成式模型的 decode 是一轮一轮迭代的，不同 request 的输出长度又不一样，所以调度粒度必须下沉到 iteration。

读 Orca 时只抓三件事：

```text
1. 为什么 request-level batching 不适合生成式模型？
2. iteration-level scheduling 解决了什么？
3. decode 阶段为什么天然会形成动态 batch？
```

第二篇读 [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)。这篇的重点不是 CUDA kernel，而是 KV cache 的生命周期管理。vLLM 把 KV cache 拆成 block，并用 block table 管理逻辑 token 到物理 KV block 的映射。这个抽象后来会反复出现：只要要做 continuous batching、prefix cache、PD transfer、KV offload，就绕不开 block/page 这层。

读 vLLM 时先看这几个概念：

```text
request lifecycle
scheduler
continuous batching
PagedAttention
KV cache block manager
block table
```

这里的关键认识是：**decode 难调度，不只是因为单步算子慢，而是因为每个 request 的 KV cache 长度、生命周期和释放时间都不同。**

## 2. 再读 PD 分离主线

我会把 PD 分离主线分成四组：Splitwise、DistServe、Sarathi、Mooncake。

### 2.1 Splitwise：为什么两个阶段适合拆

[Splitwise](https://arxiv.org/abs/2311.18677) 最适合作为 PD 分离动机的入门论文。它的主线很清楚：prefill 和 decode 的硬件资源特征不同。

你读它时应该推导这几个问题：

```text
为什么 decode GPU compute utilization 低？
为什么 decode 更关心 bandwidth / latency？
为什么 prefill 和 decode 可以使用不同卡型？
P worker 和 D worker 的比例如何由输入输出长度决定？
```

Splitwise 的价值在于把“phase splitting”这件事讲得非常朴素：如果一个阶段吃 compute，另一个阶段吃 memory bandwidth，把它们放在完全相同的机器上并不一定划算。

### 2.2 DistServe：把 PD 分离变成 SLO 优化问题

[DistServe](https://arxiv.org/abs/2401.09670) 是更系统化的 PD 分离论文。它的目标不是单纯提高 tokens/s，而是 **goodput under TTFT/TPOT SLO**。

这里有两个核心概念。

第一，colocated serving 会产生 prefill-decode interference。prefill 可能占住一大块计算时间，decode 又需要稳定地逐 token 前进。两个阶段混在一个 scheduler 里时，某些 request 的 tail ITL 会被 prefill 打断。

第二，不同阶段应该允许不同 resource allocation 和 parallelism strategy。比如 prefill 可能适合更强 tensor parallel / pipeline parallel，decode 可能更受 KV cache 容量、batch size、memory bandwidth 限制。

读 DistServe 时重点看：

```text
TTFT SLO / TPOT SLO 如何定义？
goodput 怎么算？
为什么 chunked prefill 只能缓解干扰？
placement search 的目标函数是什么？
parallelism search 如何分别服务 P 和 D？
```

### 2.3 Sarathi：PD 分离的对照组

[Sarathi](https://arxiv.org/abs/2308.16369) 和 [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) 不是 PD 分离方案，而是非常重要的对照组。它们的路线是：不把 P/D 物理拆开，而是把 prefill 切成 chunk，让 decode piggyback 在 prefill chunk 上。

所以 Sarathi 要回答的是：

```text
什么时候 chunked prefill 就够了？
chunk size 太大/太小分别有什么问题？
chunked prefill 为什么可能牺牲 TTFT？
decode-maximal batching 为什么能提高 decode 覆盖率？
```

这组论文的意义在于提醒我：**PD 分离不是唯一解。**如果 colocated 系统通过 chunked prefill 已经可以控制 tail ITL，并且 P->D KV transfer 成本很高，那么物理拆分未必更好。

### 2.4 Mooncake：从 PD 分离走向 KVCache-centric serving

[Mooncake](https://arxiv.org/abs/2407.00079) 是后面真正要深入的重点。它的核心不是“P 节点把 KV 发给 D 节点”这么简单，而是把 KV cache 变成一等公民。

Mooncake GitHub README 对架构的描述很直接：Mooncake separates prefill and decoding clusters，并利用 CPU、DRAM、SSD 等资源实现 disaggregated KVCache pool。它的核心组件包括 Transfer Engine 和 Mooncake Store，目标是把 KV cache 在 GPU、CPU、远端内存、SSD、网络之间组织起来。

读 Mooncake 时重点看：

```text
KV cache 为什么变成系统中心资源？
long context 为什么特别适合 Mooncake？
global scheduler 如何在 SLO 和 throughput 之间权衡？
early rejection 为什么在过载场景有意义？
KV cache 如何跨 GPU / CPU / SSD / NIC 管理？
```

对我来说，Mooncake 最重要的视角是：**PD 分离带来的不是一个 transfer 问题，而是一个 KV cache placement / lifecycle / scheduling 问题。**

## 3. 必须自己推导的公式

这一部分不能只看论文结论，必须自己算。PD 分离所有 tradeoff 最后都会落到 compute、memory、network 三个账本上。

### 3.1 KV cache 大小

P 到 D 真正要传的是 KV cache。对常见 decoder-only Transformer，有：

$$
\mathrm{KVBytes}=2\cdot L\cdot S\cdot H_{kv}\cdot d_h\cdot b
$$

其中：

```text
L      = layer 数
S      = token 数
H_kv   = KV head 数，GQA/MQA 下小于 attention head 数
d_h    = head_dim
b      = 每个元素字节数，FP16/BF16 = 2，FP8 = 1
前面的 2 = K 和 V
```

以 Llama-3.1-8B 这类 GQA 模型粗算，如果：

```text
L = 32
H_kv = 8
d_h = 128
b = 2
```

则每 token 的 KV cache 大小是：

```text
2 * 32 * 8 * 128 * 2 = 131072 bytes = 128 KB / token
```

如果 prompt 是 10k token，那么一个 request 的 prefill KV 就是：

```text
128 KB * 10000 = 1.28 GB
```

这就是为什么 PD 分离不是“跨节点随便传一下”。长上下文下，P->D KV transfer 很容易变成系统级瓶颈。

### 3.2 Prefill FLOPs 粗算

对 dense decoder-only Transformer，一层 prefill 可以粗略拆成：

```text
QKV/O projection: 约 8 * S * d_model^2
MLP/SwiGLU:       约 6 * S * d_model * d_ff
Attention:        约 4 * S^2 * d_model
```

总 prefill FLOPs 约为：

$$
F_{prefill}\approx L\cdot(8Sd^2+6Sdd_{ff}+4S^2d)
$$

这里有两个观察。

第一，线性层部分随输入长度 \(S\) 线性增长。第二，attention 部分随 \(S^2\) 增长，所以上下文越长，prefill 越重。但 prefill 通常可以形成较大的 GEMM，因此 GPU compute utilization 更容易打满。

### 3.3 Decode 每 token 代价

decode 每生成一个 token，大致是：

$$
F_{decode/token}\approx L\cdot(8d^2+6dd_{ff}+4S_{ctx}d)
$$

它仍然对当前上下文长度 \(S_{ctx}\) 敏感，因为每一步 attention 都要看历史 KV。但 decode 真正难的地方不只是 FLOPs：

```text
每步只生成 1 个 token
矩阵 M 维很小
权重要反复读
KV cache 要反复读
```

所以 decode 的 arithmetic intensity 通常比 prefill 差，batch 小时尤其明显。

### 3.4 P/D worker 配比

这是最应该掌握的系统公式。假设：

```text
lambda = 请求到达率，requests/s
I      = 平均输入长度，input tokens/request
O      = 平均输出长度，output tokens/request
Cp     = 单个 prefill GPU 的 prefill 能力，input tokens/s
Cd     = 单个 decode GPU 的 decode 能力，output tokens/s
Np     = prefill GPU 数
Nd     = decode GPU 数
```

稳定条件近似为：

$$
N_p C_p \ge \lambda I
$$

$$
N_d C_d \ge \lambda O
$$

所以 P/D 资源比例近似为：

$$
\frac{N_p}{N_d}\approx \frac{I/C_p}{O/C_d}
$$

这个公式给出最基本的方向感：

```text
长输入短输出：更吃 prefill
短输入长输出：更吃 decode
decode 优化后 Cd 变大：可以减少 D
prefill 并行后 Cp 变大：可以减少 P
```

真实系统还要继续加上：

```text
P->D KV transfer latency
queueing delay
TTFT SLO
ITL/TPOT SLO
KV cache memory capacity
prefix cache hit rate
network bandwidth
故障、重试、超时
```

### 3.5 P->D 传输时间

P->D 传输可以先用一个简单模型：

$$
T_{transfer}\approx \frac{\mathrm{KVBytes}}{BW_{effective}}+T_{metadata}+T_{sync}
$$

这里的 \(BW_{effective}\) 不是网卡标称带宽。它会被很多工程细节吃掉：

```text
GPU memory layout 是否连续
是否要 gather/scatter
P/D 两侧 TP size 是否一致
GQA/MLA head layout 是否一致
是否经过 CPU bounce buffer
RDMA / NVLink / TCP 后端
并发请求之间的带宽争抢
```

SGLang 文档里提到异构 TP 场景下 KV cache layout 不同，需要 GPU staging buffer：prefill 侧把 KV head slices gather 成连续 buffer，做 bulk RDMA transfer，再在 decode 侧 scatter 到正确的 KV cache pages。这个点很值得后面单独读源码。

## 4. 源码阅读顺序

### 4.1 vLLM：先看最小 1P1D

先从 vLLM 官方 [disaggregated prefill example](https://docs.vllm.ai/en/stable/examples/online_serving/disaggregated_prefill/) 开始。这个示例会启动两个 vLLM 实例，一个 prefill instance，一个 decode instance，然后在两者之间传 KV cache。

先抓住这些概念：

```text
kv_role = kv_producer / kv_consumer
kv_connector
proxy server
prefill server port
decode server port
request 如何先到 P，再到 D
P 如何只做 prefill
D 如何接着 decode
```

建议阅读路径：

```text
examples/online_serving/disaggregated_prefill.sh
benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py
vllm/distributed/kv_transfer/
vllm/distributed/kv_transfer/kv_connector/
vllm/v1/worker/gpu_model_runner.py
vllm/v1/core/scheduler.py
vllm/v1/core/kv_cache_manager.py
```

不要一开始陷进 connector 细节。先画 request 生命周期：

```text
client
  -> proxy
    -> prefill vLLM
      -> allocate KV blocks
      -> run prefill
      -> send / expose KV
    -> decode vLLM
      -> receive / read KV
      -> continue generation
```

### 4.2 NIXL / LMCache connector

第二层看 vLLM [NixlConnector](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/) 和 [LMCache disaggregated prefill quickstart](https://docs.lmcache.ai/getting_started/quickstart/disaggregated_prefill.html)。这一层的重点是 KV transfer 的工程协议。

需要盯住：

```text
connector 初始化
engine_id / role / rank
KV block address 如何注册
remote block id 如何交换
send/recv 是 push 还是 pull
metadata side channel 怎么走
失败和 timeout 怎么处理
```

我自己的通信背景会更关注：

```text
memory registration
buffer pool
scatter/gather
GPU direct or host staging
UCX env 和 NCCL env 的区别
跨 TP layout reshape
```

vLLM 文档特别提醒：NixlConnector 使用 UCX transport 时，`NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME` 这类 NCCL 环境变量不适用，应该配置 UCX 变量。

### 4.3 SGLang PD disaggregation

SGLang 是第二个必须重点看的工程实现。官方 [PD Disaggregation 文档](https://docs.sglang.ai/advanced_features/pd_disaggregation.html) 里已经说明目前支持 Mooncake 和 NIXL transfer engine，并提供 prefill server、decode server、router 的基本启动方式。

建议阅读路径：

```text
docs/advanced_features/pd_disaggregation.md
python/sglang/srt/
python/sglang/srt/managers/
python/sglang/srt/disaggregation/
python/sglang/srt/mem_cache/
python/sglang/srt/mem_cache/storage/mooncake_store/
python/sglang_router/
```

重点看：

```text
--disaggregation-mode prefill
--disaggregation-mode decode
--disaggregation-transfer-backend mooncake / nixl
router 如何选择 P/D
prefill 和 decode event loop
KV bootstrap
decode 如何等待 KV 到达
heartbeat / timeout
```

### 4.4 Mooncake

Mooncake 的工程重点是 KV cache 存储和传输。建议按组件拆开读：

```text
Mooncake Transfer Engine
Mooncake Store
SGLang Mooncake integration
metadata service / etcd
RDMA / TCP / NVLink transport
object layout
replica / striping / placement
```

读 Mooncake 不要只问“怎么传 KV”，而要问：

```text
KV object 如何命名？
一个 request 的 KV 被切成哪些 object？
object metadata 放在哪里？
多个 D worker 读同一 KV 时如何避免热点？
长上下文下如何分层存储？
什么时候存在 GPU，什么时候落 CPU/SSD？
scheduler 如何知道 KV 在哪里？
```

## 5. 该写的代码

只读论文和源码很容易觉得自己懂了，但 PD 分离是系统问题，不写模拟器很难真正有直觉。

### 5.1 项目一：PD 分离离散事件模拟器

先不要上 GPU。用 Python 写一个模拟器，输入 workload 分布，输出 TTFT、ITL、goodput。

功能最小集：

```text
请求到达：Poisson 或 trace replay
每个请求有 input_len I、output_len O
P 队列：处理 prefill
D 队列：处理 decode token iteration
P 完成后生成 KVBytes
模拟 P->D transfer
D 收到 KV 后开始逐 token decode
统计 TTFT / TPOT / E2E latency / SLO attainment
```

应该支持的参数：

```text
Np, Nd
Cp, Cd
network_bw
KV bytes per token
max batch size
chunked prefill on/off
prefix cache hit rate
```

最小类设计可以是：

```python
from dataclasses import dataclass

@dataclass
class Request:
    rid: int
    arrival: float
    input_len: int
    output_len: int
    prefill_done: float | None = None
    kv_ready: float | None = None
    first_token: float | None = None
    finish: float | None = None

@dataclass
class WorkerPool:
    n: int
    service_rate: float
    queue: list[Request]
```

事件类型先保持清楚：

```text
ARRIVAL
PREFILL_START
PREFILL_DONE
KV_TRANSFER_DONE
DECODE_STEP_DONE
REQUEST_DONE
```

最后要能画出：

```text
Np/Nd 比例变化时 TTFT 和 ITL 怎么变
input_len 增大时 P 是否成为瓶颈
output_len 增大时 D 是否成为瓶颈
network_bw 下降时 PD 什么时候反而变差
```

### 5.2 项目二：最小 1P1D Fake KV Transfer

第二个项目写三个进程：

```text
proxy.py
prefill_server.py
decode_server.py
```

流程：

```text
client -> proxy
proxy -> prefill_server
prefill_server fake 生成 KV tensor
prefill_server 把 KV 写入共享内存 / TCP socket / mmap file
proxy -> decode_server
decode_server 读取 KV
decode_server fake decode 输出 token
```

fake KV 可以先用：

```python
kv = torch.randn(
    num_layers,
    2,
    num_tokens,
    num_kv_heads,
    head_dim,
    dtype=torch.float16,
)
```

必须自己定义 metadata：

```text
request_id
num_layers
num_tokens
dtype
shape
storage_key
block_ids
checksum
```

这个项目的目的不是模拟模型，而是暴露真实系统最容易出错的地方：

```text
KV layout 不一致
token offset 不一致
layer index 不一致
dtype 不一致
block id 映射错误
P 还没传完，D 已经开始读
timeout 后资源泄露
```

### 5.3 项目三：跑 vLLM 1P1D

参考 vLLM 官方示例，先单机两张卡跑：

```text
baseline: 单 vLLM
PD: 1P1D
```

实验矩阵：

```text
输入长度: 512 / 2k / 8k / 16k
输出长度: 128 / 512 / 2k
并发: 1 / 4 / 16 / 64
指标: TTFT, ITL, E2E latency, throughput, GPU util, HBM usage, PCIe/NVLink/RDMA traffic
```

观察重点不是“PD 一定更快”，而是：

```text
短输入短输出：PD 可能不划算
长输入短输出：P 可能成为瓶颈
短输入长输出：D 可能成为瓶颈
长输入长输出：P/D 和 KV 传输都会很重
```

### 5.4 项目四：跑 SGLang + Mooncake / NIXL

先小模型，不要一上来 DeepSeek-V3：

```text
Qwen2.5-7B 或 Llama-3.1-8B
单机 2 GPU：1P1D
单机 4 GPU：1P + 1D，各自 TP=2
单机 4 GPU：P TP=1 多副本，D TP=2/4
跨节点：P/D 分离 + RDMA
```

重点验证：

```text
Mooncake backend vs NIXL backend
TCP vs RDMA vs NVLink
同构 TP vs 异构 TP
staging buffer on/off
高并发下 KV transfer timeout
D worker cache saturation
```

## 6. 性能分析 checklist

不要直接 profile 整个 PD 系统。先把 prefill 和 decode 拆开。

Prefill-only：

```text
input_len = 512 / 2k / 8k / 16k
max_tokens = 1
```

Decode-heavy：

```text
input_len = 128 / 512
output_len = 1k / 4k
```

Prefill 看：

```text
GPU SM utilization
Tensor Core utilization
每层 GEMM 时间
attention 时间随 S 的增长
TTFT 分解：
  queue time
  prefill compute time
  KV transfer time
  first decode step time
```

Decode 看：

```text
ITL / TPOT
decode batch size
每 token HBM read
KV cache read bandwidth
weight read bandwidth
scheduler idle gap
D worker 是否等 KV
```

P->D transfer 看：

```text
KVBytes/request
effective bandwidth
transfer queue length
metadata latency
RDMA / NVLink / TCP backend
是否有 host bounce
是否有 gather/scatter
失败重试 / timeout
```

## 7. 8 周学习路线

| 周数 | 读什么 | 写什么 | 目标 |
| --- | --- | --- | --- |
| 第 1 周 | Orca、vLLM / PagedAttention | continuous batching 小模拟器 | 懂 request/iteration 调度 |
| 第 2 周 | Splitwise、DistServe | P/D 配比计算脚本 | 懂为什么分离 |
| 第 3 周 | Sarathi、Sarathi-Serve | chunked prefill 模拟器 | 懂 PD 的替代方案 |
| 第 4 周 | Mooncake 论文 | KV cache 大小/传输成本计算器 | 懂 KVCache-centric |
| 第 5 周 | vLLM disaggregated prefill | Fake 1P1D KV transfer | 懂协议和生命周期 |
| 第 6 周 | vLLM KV connector / NIXL | 跑 vLLM 1P1D benchmark | 懂真实 connector |
| 第 7 周 | SGLang PD docs/source | 跑 SGLang 1P1D | 懂 router、P/D worker |
| 第 8 周 | Mooncake Store/source | 加 KV store / cache hit 模拟 | 懂生产级 KV 管理 |

## 8. 最终知识地图

学完这个主题后，脑子里应该有这张图：

```text
                        +--------------+
client requests ------->| router/proxy |
                        +------+-------+
                               |
                +--------------+--------------+
                |                             |
        +-------v--------+            +-------v--------+
        | Prefill workers |            | Decode workers |
        | compute-bound   |            | memory/latency |
        | TTFT-sensitive  |            | ITL-sensitive  |
        +-------+--------+            +-------^--------+
                |                             |
                | KV cache                    |
                v                             |
        +-------------------------------------+
        | KV transfer / KV store              |
        | NIXL / Mooncake / LMCache           |
        | NVLink / RDMA / TCP / CPU / SSD     |
        +-------------------------------------+
```

PD 分离的本质不是“把服务拆成两个进程”，而是：

```text
把 prefill compute、decode iteration、KV cache 生命周期、网络传输、调度策略解耦。
```

## 9. 每读一篇都问自己的 5 个问题

```text
1. 这个系统里 TTFT = 哪几段时间相加？
2. 这个系统里 ITL = 哪几段时间相加？
3. 一个 request 的 KV cache 到底有多大？
4. P worker 和 D worker 的最优比例怎么估？
5. P->D 传输变慢时，PD 分离是不是反而输给 colocated/chunked prefill？
```

这 5 个问题如果能推明白，PD 分离基本就入门了。

## 参考文献和源码入口

### 必读论文

| 优先级 | 论文 | 重点 |
| --- | --- | --- |
| 1 | [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | iteration-level scheduling / selective batching |
| 2 | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | KV block / block table / PagedAttention |
| 3 | [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference](https://www.usenix.org/conference/osdi24/presentation/agrawal) | chunked prefill + stall-free scheduling |
| 4 | [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) | phase splitting / hardware heterogeneity |
| 5 | [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670) | TTFT / TPOT SLO / goodput / placement search |
| 6 | [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) | KVCache-centric architecture / disaggregated KV cache |
| 7 | [P/D-Serve: Serving Disaggregated Large Language Model at Scale](https://arxiv.org/html/2408.08147v1) | 生产级 P/D serving、动态比例、KV transfer 优化 |

### 源码和官方文档

| 优先级 | 项目 | 入口 |
| --- | --- | --- |
| 1 | vLLM disaggregated prefilling | [feature doc](https://docs.vllm.ai/en/stable/features/disagg_prefill/) / [example](https://docs.vllm.ai/en/stable/examples/online_serving/disaggregated_prefill/) |
| 2 | vLLM NixlConnector | [NixlConnector Usage Guide](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/) |
| 3 | LMCache disaggregated prefill | [quickstart](https://docs.lmcache.ai/getting_started/quickstart/disaggregated_prefill.html) |
| 4 | SGLang PD Disaggregation | [official doc](https://docs.sglang.ai/advanced_features/pd_disaggregation.html) |
| 5 | Mooncake | [GitHub](https://github.com/kvcache-ai/Mooncake) |
| 6 | DistServe repo | [GitHub](https://github.com/LLMServe/DistServe) |
| 7 | Splitwise simulator | [GitHub](https://github.com/Mutinifni/splitwise-sim/blob/main/README.md) |

### 案例文章

| 文章 | 重点 |
| --- | --- |
| [Splitwise improves GPU usage by splitting LLM inference phases](https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/) | Splitwise 作者团队博客，适合快速建立 phase splitting 直觉 |
| [Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) | SGLang / DeepSeek / PD disaggregation / large-scale EP 的生产案例 |
| [Mooncake Docs](https://kvcache-ai.github.io/Mooncake/) | 跟踪 Mooncake 与 vLLM、SGLang、LMCache、NIXL 的集成动态 |
