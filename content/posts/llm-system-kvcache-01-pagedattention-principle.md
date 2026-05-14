---
date: '2026-05-10T12:46:00+08:00'
draft: false
title: 'LLM System: KV Cache 查询 01 - PagedAttention 原理'
categories: ["AI"]
tags: ["LLM", "LLM System", "Serving", "KV Cache", "PagedAttention"]
series: ["LLM System", "KV Cache Lookup"]
series_order: 1
weight: 2
math: true
---

> TODO: 这里写 PagedAttention 的核心抽象：block/page、block table、逻辑 token 到物理 KV block 的映射。

## 基础：tensor 级拆请求的形状（大量细节）

定义符号：$B$ 是 batch size，$T$ 是 seq_len，$D$ 是 token_dim，$d_q$ 是把 embedding token 投影到 $Q$ 后的维度。

推理框架拿到的请求是：\(R \in \mathbb{R}^{B \times T}\)。

$R_{b,t}$ 是一个最最基本的 token id 标量。

raw 请求经过 embedding lookup，做的操作是把这个 token 标量映射成一个高维向量。假设原先 token 是 `1234` 这个标量，现在就把 token 映射成 `[0.1, 0.2, 0.3, 0.4]` 这样的向量。

所以 $R$ 经过 embedding lookup 之后，得到：\(X \in \mathbb{R}^{B \times T \times D}\)。

因为我们目前只考虑推理场景，所以把 $W_Q$、$W_K$、$W_V$ 之类的矩阵当成固定的模型参数。

然后很多博客会直接写：\(Q = XW_Q\)。

这样写很容易产生误解，因为这好像是把整个原始请求丢给了 $Q$ 去投影。从感觉上这也没问题，但是对深入理解帮助比较小。

实际上注意力投影的最小粒度是每个 token embedding vector，也就是 \(X_{b,t,:}\)。

所以在最小粒度的视图下，$x$ 的形状是 \(x \in \mathbb{R}^{D}\)。

如果没有多头注意力，$W_Q$ 的形状就是 \(W_Q \in \mathbb{R}^{D \times d_q}\)。

在最常见的单头简化设定下，可以取 $d_q = D$。为了让 $q$ 和 $k$ 能做点积，通常需要 $d_q = d_k$。

那么投影后，对于每个 token 的 $q$ 就变成了 $[d_q]$ 形状。$K/V$ 也是同理。单头下最小的逐 token 操作粒度如下：

公式视角是：\(q = xW_Q\)，\(k = xW_K\)，\(v = xW_V\)。

```text
q = x @ W_Q   [d_q]
k = x @ W_K   [d_k]
v = x @ W_V   [d_v]
```

每个 embedding token 被投影成三个向量 $q$、$k$、$v$。

将其扩展到整个 batch + 整个 seq：

公式视角是：\(Q = XW_Q\)，\(K = XW_K\)，\(V = XW_V\)。

```text
Q = X @ W_Q   [B, T, d_q]
K = X @ W_K   [B, T, d_k]
V = X @ W_V   [B, T, d_v]
```

这里“扩展”的意思是：对于每一个 batch index $b$ 和每一个位置 $t$，都执行一次同样的逐 token 投影。固定某个 $b,t$ 后，投影结果是：

公式视角是：\(Q_{b,t,:} = X_{b,t,:}W_Q\)，\(K_{b,t,:} = X_{b,t,:}W_K\)，\(V_{b,t,:} = X_{b,t,:}W_V\)。

```text
Q[b, t, :]   [d_q]
K[b, t, :]   [d_k]
V[b, t, :]   [d_v]
```

写成代码就是：

```python
for b in range(B):
    for t in range(T):
        Q[b, t, :] = X[b, t, :] @ W_Q
```

这里说明了一件事情：对于不同 batch 且不同位置的 token，都是用同一个 $W_Q$ 去投影的。

这是合理的，因为 $W_Q$ 表示的是这一层学到的通用投影规则。一个 embedding token 如何被投影，本质上取决于当前层这个位置的 hidden vector $X_{b,t,:}$ 和该层共享的 $W_Q$。

batch index $b$ 本身不携带语义，它只是并行计算时的编号。真正决定“这是哪个请求”的，是 $X_{b,:,:}$ 里的 token 内容。

position index $t$ 也不是通过换一套 $W_Q$ 来发挥作用的。位置信息通常已经通过 position embedding、RoPE、causal mask，或者前面层的上下文化 hidden state 体现在 $X_{b,t,:}$ 或 attention 计算里。

所以我们不需要给不同 batch 或不同位置各自准备一套 $W_Q$。否则参数量会随着 batch size 或 sequence length 增长，而且会破坏 Transformer 对不同位置共享同一套 token 处理规则的设计。

在计算注意力分数的时候，依然从逐 token 去看：如定义所述，就是让一个 batch 内的第 $i$ 个 token 的 $q$ 和第 $j$ 个 token 的 $k$ 做一次乘法。所以注意力分数矩阵应该是如下形状：

```text
[B, T, T]
```

现代的模型有很多都用了 GQA。GQA 的改动简单来说就是让一个 token 被多个 $Q$ 和多个 $KV$ 去投影。$Q$ 的数量是 $KV$ 的整数倍。既然有整数倍就有组的对应，这里不做赘述。

定义符号：$H_Q$ 是 $Q$ 的组数，$H_{KV}$ 是 $KV$ 的组数。

相应也得修改 $W_Q/W_K/W_V$ 的形状。这里有一个非常重要的工程细节：线性层投影出来的结果，通常不是直接得到 `[head_num, head_dim]` 的二维结构，而是先得到一个扁平的一维向量，然后再 reshape 成 head 形式。

从逐 token 视角看：

公式视角是：\(W_Q \in \mathbb{R}^{D \times (H_Q d_h)}\)，\(W_K \in \mathbb{R}^{D \times (H_{KV} d_h)}\)，\(W_V \in \mathbb{R}^{D \times (H_{KV} d_h)}\)。

```text
x: [D]

W_Q: [D, H_Q  * d_h]
W_K: [D, H_KV * d_h]
W_V: [D, H_KV * d_h]
```

所以先得到 raw 投影结果：

公式视角是：\(q_{\text{raw}} = xW_Q\)，\(k_{\text{raw}} = xW_K\)，\(v_{\text{raw}} = xW_V\)。

```text
q_raw = x @ W_Q   [H_Q  * d_h]
k_raw = x @ W_K   [H_KV * d_h]
v_raw = x @ W_V   [H_KV * d_h]
```

这里 `[H_Q * d_h]` 中间是乘法，表示这是一个长度为 $H_Q \times d_h$ 的一维向量。它还没有显式拆成 head 维度。

然后再 reshape：

公式视角是：\(q = \operatorname{reshape}(q_{\text{raw}}, [H_Q, d_h])\)，\(k = \operatorname{reshape}(k_{\text{raw}}, [H_{KV}, d_h])\)，\(v = \operatorname{reshape}(v_{\text{raw}}, [H_{KV}, d_h])\)。

```text
q = reshape(q_raw, [H_Q,  d_h])
k = reshape(k_raw, [H_KV, d_h])
v = reshape(v_raw, [H_KV, d_h])
```

也就是：

```text
q: [H_Q,  d_h]
k: [H_KV, d_h]
v: [H_KV, d_h]
```

所以 `[H_Q * d_h]` 和 `[H_Q, d_h]` 的元素总数相同，但含义不同：

```text
[H_Q * d_h] 是一个扁平向量；
[H_Q, d_h] 是已经拆成 H_Q 个 head、每个 head 是 d_h 维的二维结构。
```

$q$、$k$、$v$ 也都变成了一组向量。但这个时候 $q$ 和 $kv$ 的向量形状不一定相等了。

计算注意力分数的时候，注意力矩阵从原来的 $[B,T,T]$ 变成：

```text
[B, T, T, H_Q]
```

注意，这里计算完 $KV$ 之后，缓存的 KVCache 其实就是：

```text
k: [H_KV, d_h]
v: [H_KV, d_h]
```

如果是 MLA，还有一些变化。这里不纠结具体的数学原理，只看关键的形状部分。

MLA 不直接缓存 $k$ 和 $v$。

还是从逐 token 的视角去看。对于一个 token：

公式视角是：\(x = X_{b,t,:}\)。

```text
x = X[b, t, :]
x: [D]
```

先得到 KV latent：\(c_{kv} = xW_{DKV}\)。

其中：

```text
W_DKV: [D, d_c]
c_kv:  [d_c]
```

所以 MLA 推理时核心缓存的是：

```text
C_KV_cache per token: [d_c]
```

扩展到 batch + seq：

```text
C_KV_cache: [B, T, d_c]
```

如果为了理解，把 latent 显式还原成 $K/V$，可以写成：

公式视角是：\(k_{\text{raw}} = c_{kv}W_{UK}\)，\(v_{\text{raw}} = c_{kv}W_{UV}\)。

```text
k_raw = c_kv @ W_UK   [H_Q * d_h]
v_raw = c_kv @ W_UV   [H_Q * d_h]

W_UK: [d_c, H_Q * d_h]
W_UV: [d_c, H_Q * d_h]
```

注意这里同样是先得到扁平向量 `[H_Q * d_h]`，再 reshape：

公式视角是：\(k = \operatorname{reshape}(k_{\text{raw}}, [H_Q, d_h])\)，\(v = \operatorname{reshape}(v_{\text{raw}}, [H_Q, d_h])\)。

```text
k = reshape(k_raw, [H_Q, d_h])
v = reshape(v_raw, [H_Q, d_h])
```

所以：

```text
k: [H_Q, d_h]
v: [H_Q, d_h]
```

$Q$ 可以先简化理解为仍然从 $x$ 直接投影出来：

公式视角是：\(q_{\text{raw}} = xW_Q\)，\(q = \operatorname{reshape}(q_{\text{raw}}, [H_Q, d_h])\)。

```text
q_raw = x @ W_Q   [H_Q * d_h]
q = reshape(q_raw, [H_Q, d_h])
```

于是注意力分数仍然是：

```text
[B, T, T, H_Q]
```

其实这里可以把 GQA 纳入到 MLA 的框架中去，因为 MLA 是通过 $W_{UK}$ 和 $W_{UV}$ 两个矩阵去从 $C_{KV}$ 里面还原出 $KV$，这是基于一个假设：$C_{KV}$ 有能力保存 $K$ 和 $V$ 的低秩压缩。

那么假设我们这里将 $C_{KV}$ 变得足够大，使其刚好等于 $K$ 和 $V$，并且让 $W_{UK}$ 和 $W_{UV}$ 矩阵仅做选择功能，这个时候，只要我们对应好 $Q$ 和 $KV$ 的 group 关系，其实也能在 MLA 的一套框架下实现出一个 GQA。我希望后续的工程设计上能统一这部分的代码设计。

## 基础：自回归时的数据流形态

用语言描述这个过程很简单。笔者这里以某一层为例：prefill 阶段计算注意力和缓存 KV，得到一个 `[B, T, T, H_Q]` 的注意力分数矩阵和两个 `[B, T, H_KV, d_h]` 的 KV 矩阵，并且取计算最后一个 token 时的输出 logits 作为新的 token。

decode 阶段则是重复做如下操作：

```text
计算新 token qkv
-> 写 kvcache
-> 计算注意力分数
-> 写注意力分数
-> 取模型输出的 logits
```

直到输出 EOF。

如果要考虑多层的话，attention score 和 KVCache 的存储 tensor 形状还要加一个 layer 的维度。

这里还有一个需要强调的点，就是工程实现上不会直接保存 `[B, T, T, H_Q]` 的注意力分数矩阵，因为有 FA、PA 这种优化在。

所以每一步 decode，数据和流向就是这样：

```text
input token:
X_new: [B, 1, D]

output qkv:
Q_new: [B, 1, H_Q,  d_h]  -- 用于计算 logits
K_new: [B, 1, H_KV, d_h]  -- 写入 kvcache
V_new: [B, 1, H_KV, d_h]  -- 写入 kvcache
```

公式视角是：\(Q_{\text{new}} = X_{\text{new}}W_Q\)，\(K_{\text{new}} = X_{\text{new}}W_K\)，\(V_{\text{new}} = X_{\text{new}}W_V\)。

如果是 MLA，$k$ 和 $v$ 会被合并成一个东西。所以工程上可以考虑把 `abstract kv cache info` 作为一个抽象的实现基类，而多态体现在如何得到 `abstract kv cache info` 上。

但是笔者目前除了 GQA 系列和 MLA，还没看过其他的 QKV 映射方案。所以这部分的设计感觉还是不够 general，等待后续增加见识后确定如何设计再动手优化工程设计。

此外 MLA还涉及很多复杂的实现细节，比如QK的hiddendim需要加上rope，而v不需要。此外还有MLA的各种变体，例如TPA，MFA等。这些都要等笔者了解之后，思考一下MLA类注意力的kv是在什么地方做创新的，再去给出最好的工程实现。这个部分会单开一部分讲。

## paged attention原理

前面铺垫了很久，到这里解释起来就非常容易。
前面说了，生成出来的KV Tensor形状是[L,B,T,HKV,DH]，但是这只是理想情况，考虑一下实际场景会发生什么。我们可以重新审视这个形状，KV Tensor的形状过于规整了，这是因为理想中这个KV Tensor的每个batch的每个请求都有着相同的T。但实际上并不是。如果静态按照这个形状去分配，显然会很浪费空间。那么很自然就会想到动态扩容。也就是给每个request一个很小的T_init，如果不够长再去分配。但这个代码写出来效率不高。

伪代码大概要做如下几步：
    if len == capacity
        new_k = allocate(capacity*factor)
        new_v = allocate(capacity*factor)

        copy old kv to new kv
        free old kv

        crud处理其他引用

连续的allocate在大容量的时候会很费时间。而且factor也不好选择。
pa的办法是不以一个KV为分配单位，而是以KV_block_size大小分配。如果一个请求的token长度是68，block_size的大小是32，则消耗3个block。这样最多浪费一个block的空间。如果只看某一层的 K cache 或 V cache，一个 block 的形状可以理解为 [block_size, HKV, DH]，元素数是 block_size * HKV * DH。K 和 V 两份合起来就是 2 * block_size * HKV * DH。（所以为什么不叫blockattention呢）pa只有划分block还不够，pa还允许了这些block之间在物理上可以不连续，而只需要一个table保存他们之间的顺序即可。

核心思路说完了，但是还是有一些实现的细节。

还是从第一性原理出发，既然推理是在做forward，也就是正常计算attention。而计算attention时，写kernel的人（更深入点说是cuda/triton & 驱动）需要从某个地方拿到KVCache。cpu侧的推理框架知道当前计算上下文对应的req拥有哪些gpu block来存放kv。最朴素的做法就是在launch kernel之前把kv table的信息传给kernel。

如果用的是cuda，传table的方式是使用一个memcpy同步table。
一个stream上执行的任务依次为：(cudaMalloc, 可能会有，就看是否做池化了)->memcpyH2D->Launch Kernel。
（补充代码实例）

如果用的是triton，传table的方式比较无痛，假设cpu的tensor已经被初始化了：t = torch.tensor([100,100])，直接做t.to('cuda')就行。内部都会被torch封装好。（补充代码实例）

如果用的是别的DSL呢？笔者用GPT5.5做了一番调研，发现可以做异构计算的DSL非常非常多。而且不是所有的DSL都和cuda这套event-stream-async copy的抽象模型一样。

笔者给这里的h-d meta data transfer模型分成三类：1. 类cuda 2. 类torch 3. 状态声明式
类cuda的例子包括cuda，hip，opencl，ascendC，这类的共性是都有一个任务队列orStream，拷贝和launch都是异步的。用event轮询，做同步。
类torch的例子包括torch，triton, xla/jax，tensorRT，这类的共性是不暴露host-device交互本身的流程。但实际上相当于一种回调函数的语法糖。
状态声明式的例子包括openmp, sycl，和类torch不同的是，这里需要device侧声明需要什么buffer，由底层自动在runtime时期调度，决定何时将信息呈递给kernel。

所以笔者在这里认为，如果想设计一个好的推理框架，不应该只把h-d meta data transfer模型设计成上述三者中的一种，而是要足够兼容这三种模型，这样推理框架能适配不同的后端。

大概想了一下，基本上只需要适配这几个通用接口，backend自己实现去就行了：
声明metadata的属性（包括size，type，所属权，backend-tensor类型等）
host_acquire  = host/CPU 侧准备访问 metadata
device_acquire = device/GPU 侧准备访问 metadata
host_release  = host/CPU 侧访问结束
device_release = device/GPU 侧访问结束
实际用的时候，用法就是：
host_acquire(op=read)
read()
host_release()
device_acquire(op=write)
write()
write_release()
看似很完美，但是@Fain（https://github.com/Koas-W）给笔者指出了一个设计上的致命错误，导致笔者放弃了做这种统一前端的设计。这种统一前端只能用于eager模式，但是不能适配cuda graph。（且不说其他backend是否支持graph，假设都支持也是会有很大问题的）

原因是这样的，如果我们做语义上的backend无感知抽象，那acquire和release必然是一个动态的语义。因为不同backend的数据驻留形式不同，有的会在cpu-gpu都驻留数据，有的则是二选一驻留。因此在做同步的时候，抽象回调函数实际链接到的的后端接口需要判断驻留的模式，这就天然不适配cuda graph。因为上述的判断逻辑就形成了一种if分支，这个刚好是cuda graph最讨厌的。cuda graph在capture了之后，每次都执行一样的操作。此外还有一个问题，cpu-gpu的数据可能不同步，这就需要维护一个dirty位。基于dirty bit的判断也是对graph不友好的。

当然上述是具体讲了一个case描述，从原理来讲也是这样的，一切这种兼容动态backend的设计都和cuda graph天生不兼容，例如nccl的多transport后端。我们的项目里如果想支持eager模式，用这种动态backend是ok的，但是如果是graph模式就不行了。为了性能，那还是一开始就支持graph吧。

似乎有点跑题了-.-

