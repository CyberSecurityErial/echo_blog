---
date: '2026-06-18T00:00:00+08:00'
draft: false
title: 'LLM System: 训练框架随笔 06 - Megatron Dataset'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Training", "Training Framework", "Megatron", "Dataset"]
series: ["LLM System", "Training Framework Notes"]
series_order: 6
weight: 1
math: true
---

## mcore dataset构造协议

不同模型对于“sample”的定义不同，也就是说每一类模型拿来训练的输入是不一样的，哪怕底层数据是一样的。所以就有必要在之上抽象一层dataset类。

主流这三类：

GPT

数据-标签形态：

tokens = x[0 : L]

labels = x[1 : L+1]

任务：

预测下一个tk

BERT

数据-标签形态

input  = corrupt(x)

labels = x only at masked positions

任务：

挖空填词，看到左右的tk预测中间的tk

T5

数据-标签形态

encoder_input = corrupt_span(x)

decoder_label = removed_spans

任务：

encoder看残缺文本，decoder输出删掉的整个片段

## mcore dataset涉及到的系统级优化

### 用mmap映射token

mmap是把一个文件映射到了进程的虚拟地址空间，但是不会把整个文件load到进程的内存。真正的读取行为是lazy的，用哪块读哪块。（实际上os做的事情就是读va-查询页表-pagefault触发-os做一次read SSD到mem，信息是mmap给-更新页表）为什么不用read()而是使用mmap()呢，按理说rea也可以做到这种lazy的行为。因为read()在读的时候不是0拷贝。

llm训练数据量大，连续存储，每次只用一块mb，适合用mmap。

mmap存在的代价是A。容易缺页，（读取没有时空局部性会这样）B。warmup慢，C。多rank同时缺页的时候os的fs开销大（为什么，因为）。

对A问题，mgt做的优化：

document_index

sample_index

shuffle_index

cache

访问密度判断

对C问题，mgt做的优化：

cache 预建

defer mmap

fast cache load

object storage block cache

先不展开。让ai总结了一下这几个优化所在的位置。方便后续索引。

具体位置在索引AAA这一节。

nv文档提了几个这部分优化的点：

关于启动瓶颈：

nv文档里面具体说了关于大规模load数据的时候容易产生的系统问题：256节点以上，真正的瓶颈在于真正读mb之前，megatron构建读取索引的过程和barrier同步。这部分构件索引主要是文件系统的开销，而barrier则是先让某些代表rank去构建idx，然后其他rank等着，直到代表rank构建完毕才能继续，显著影响整体利用率。

关于数据的shuffle：

因为训练算法上是希望通过随机性去训练/验证泛化性，但是系统上追求的是局部性和确定性，这是一对tradeoff，为了解决这个tradeoff mgt搞了一个shuffleidx相当于做了个确定化的随机，或者说给随机行为提前打表了。

关于启动之后的瓶颈：

因为刚才说了计算idx需要所有rank都barrier，因为不等到idx没办法取对的数据。barrier结束之后就会导致所有rank几乎同时走到mmap阶段了，刚才说过由于首次缺页，mmap的overhead比较大，所以可能会导致OS的fs io爆炸。

关于解决办法：

nv希望离线构建dataset idx，用的脚本是tools/prepare_cache.py。然后绕过runtime生成idx这套启动路径：--dataloader-fast-cache-load。

另外还有一种办法就是让mmap本身都变成lazy的，这样不会io burst，用这个参数：--dataloader-defer-npy-index-mmap

### shuffle index

其实上面不小心把shuffle idx的优化说完了。也就是针对mmap带来的一点问题。所以提前计算这个idx然后根据idx做prefetch。

### index dtype优化

按index范围选择尽量短的数据类型，index会存在ssd，还会被mmap到memory，而且要给其他rank共享，所以有必要节省空间。

token的dtype一般是uint16或int32

sampleidx和shuffleidx一般都是int32，太多会上到int64

GPTDataset 的 index 量级可以直接用 sample 数估算：设训练 token 数为 T、序列长度为 S，则样本数 N≈T/S。MCore 的 `sample_index` 形状是 `[N+1, 2]`，通常用 int32，所以约占 `8N` bytes；`shuffle_index` 形状是 `[N]`，通常用 uint32，所以约占 `4N` bytes；两者合计约 `12N` bytes。也就是说，seq_length=4096 时，1.2T tokens 大约有 2.93 亿个 sample，对应 `sample_index + shuffle_index` 约 3.5GB；18.5T tokens 大约有 45.2 亿个 sample，对应约 54GB。除此之外还有 `document_index`，大小约为 `4 × document_count × epoch_count` bytes，所以 document 特别多的数据集还会额外放大 index 体积。简而言之，1T token 级别的 Megatron index 通常是几 GB，10T+ token 级别会到几十 GB，30T token 级别可能接近百 GB；这也是为什么 MCore 要专门做 index cache、defer mmap 和 fast cache load。

### c++加速构建idx

第四个优化是 MCore 把 sample_index、blend_index、BERT/T5 mapping 这类超大数组的构建放到 C++ helpers.cpp 里做。具体实现先不看。

### 对象存储优化

本地的文件系统可以直接mmap，但是如果是远端的用S3管理的对象存储，不能用mmap，只能用HTTP GET去请求，为了减少请求数量，会在memory cache miss（也就是需要读取对象存储了）的时候发一个比较大的HTTP Range GET，一次取一大块，减少发请求的数量。

### dp优化，减少一次idx广播通信

DP rank 的样本切分不需要通信。所有数据并行 rank 共享同一个 dataset、shuffle 顺序和 consumed_samples；每个 rank 只用 data_parallel_rank、data_parallel_size、micro_batch_size 做整数切片，就能算出自己负责的 sample id 区间。这样避免了广播 sample ids，也让 checkpoint resume 只需要恢复 consumed_samples，就能重新对齐所有 DP rank 的数据进度。

### nv文档来源。

### 索引AAA

#### 1. `.bin` mmap 读取

* 文件：`megatron/core/datasets/indexed_dataset.py`
* 关键类：`_MMapBinReader`
* 关键逻辑：`IndexedDataset.initialize()`
* 关键词：`mmap`、`numpy.frombuffer`
* 作用：`.bin` 不整体读入进程内存，而是通过 mmap 按需访问。

---

#### 2. 三层样本路由索引

* 文件：`megatron/core/datasets/gpt_dataset.py`
* 构建函数：`GPTDataset._build_document_sample_shuffle_indices()`
* 查询函数：`GPTDataset._query_document_sample_shuffle_indices()`
* 三个索引：

  * `document_index`
  * `sample_index`
  * `shuffle_index`
* 路径：

  * `sample_id → shuffle_index → sample_index → document_index → .bin offset`

---

#### 3. index cache

* 文件：`megatron/core/datasets/gpt_dataset.py`
* 函数：`GPTDataset._build_document_sample_shuffle_indices()`
* 缓存文件：

  * `description.txt`
  * `document_index.npy`
  * `sample_index.npy`
  * `shuffle_index.npy`
* 关键词：`path_to_cache`、`numpy.save`、`numpy.load`
* 作用：索引构建一次，后续复用。

---

#### 4. deferred index mmap

* 文件：`megatron/core/datasets/gpt_dataset.py`
* 构建侧：`GPTDataset._build_document_sample_shuffle_indices()`
* 查询侧：`GPTDataset._query_document_sample_shuffle_indices()`
* 参数：`--dataloader-defer-npy-index-mmap`
* 关键词：`defer_npy_index_mmap`、`mmap_mode="r"`
* 作用：启动时不立刻 mmap 三个 `.npy` 索引，第一次访问样本时再 lazy load。

---

#### 5. fast cache load

* 文件：

  * `megatron/core/datasets/gpt_dataset.py`
  * `megatron/core/datasets/indexed_dataset.py`
* 参数：`--dataloader-fast-cache-load`
* 关键词：`fast_cache_load`
* 作用：假设 index cache 已经存在，跳过部分 cache 检查和启动同步路径，加快大规模训练启动。

---

#### 6. cache 预建

* 文件：`tools/prepare_cache.py`
* 典型参数：

  * `--data-cache-path`
  * `--global-batch-size`
  * `--seq-length`
  * `--prepare-cache-world-size`
* 作用：训练前离线构建 GPT dataset cache，避免训练启动时 rank 0 现建、其他 rank 等待。

---

#### 7. 访问密度判断

* 文件：`megatron/core/datasets/gpt_dataset.py`
* 函数：`GPTDataset._build_document_sample_shuffle_indices()`
* 关键词：

  * `len(document_index) * 2 > len(self.dataset.sequence_lengths)`
  * `sequence_lengths.copy()`
* 注意：这是构建 `sample_index` 时对 `sequence_lengths` 的优化，不是运行时 sample 读取优化。
* 作用：当 mmap 数组访问密度很高时，主动 copy 到 RAM，避免大量随机 page fault。

---

#### 8. C++ 构建大索引

* 文件：

  * `megatron/core/datasets/helpers.cpp`
  * `megatron/core/datasets/helpers.py`
* 典型函数：

  * `build_sample_idx`
  * `build_blending_indices`
* 作用：把数亿级顺序循环放到 C++，避免 Python 循环开销。

---

#### 9. object storage index cache

* 文件：

  * `megatron/core/datasets/object_storage_utils.py`
  * `megatron/core/datasets/indexed_dataset.py`
* 关键函数：

  * `get_index_cache_path()`
  * `cache_index_file()`
  * `IndexedDataset.__init__()`
* 关键词：`object_storage_cache_path`
* 作用：对象存储上的 `.idx` 先下载到本地 cache，再由本地 index reader 使用。

---

#### 10. S3 `.bin` block cache / range read

* 文件：`megatron/core/datasets/indexed_dataset.py`
* 关键类：`_S3BinReader`
* 配置：`ObjectStorageConfig.bin_chunk_nbytes`
* 默认块大小：`256 * 1024 * 1024`
* 关键词：

  * `get_object`
  * `Range=f"bytes=..."`
  * `_cache`
  * `_cache_bytes_start`
  * `_cache_bytes_end`
* 作用：S3 上的 `.bin` 不整体下载；按 range 读取大块，并在 Python 侧缓存当前 block。

---

#### 11. MSC byte-range read

* 文件：`megatron/core/datasets/indexed_dataset.py`
* 关键类：`_MultiStorageClientBinReader`
* 关键词：`byte_range`
* 注意：MSC 路径不是 `_S3BinReader` 那套 Python block cache，而是交给 Multi-Storage Client 做 byte-range read。
