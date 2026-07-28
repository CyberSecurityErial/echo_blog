---
date: '2026-07-28T20:55:18+08:00'
draft: false
title: 'LLM System: SGLang 01 - 共享专家融合'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "SGLang", "MoE", "Shared Expert", "Kernel Fusion", "Expert Parallelism"]
series: ["LLM System", "SGLang"]
series_order: 1
weight: 1
math: false
---

> 这篇笔记把 SGLang 共享专家融合的计算语义、权重布局和执行路径压缩成六张结构图。图中的实现判断以 2026-07-29 的 SGLang `main` 分支为准；正文只补充图中容易误读的部分。

## 从双路径到一次 MoE GEMM

![SGLang 共享专家融合：从双路径到一次 MoE GEMM](/images/llm-system-sglang-shared-expert-fusion/01-fusion-overview.png)

融合不是让 Router 在 Shared Expert 与 Routed Expert 之间做选择。Shared Expert 仍然是必经计算，只是被追加为一个额外的 Expert Slot，并随路由结果一起交给 MoE Kernel。这样 Routed Expert 与 Shared Expert 可以在一次 MoE GEMM 中完成，省去独立的 Shared Expert GEMM 与 Kernel Launch。

## Layout、Top-K 与权重重映射

![SGLang 共享专家融合的 Layout、Top-K 与权重重映射](/images/llm-system-sglang-shared-expert-fusion/02-layout-topk-weight-remap.png)

普通布局是在 `N` 个 Routed Expert 后追加一个 Shared Expert，得到 `N + 1` 个 Slot，并把实际执行的 Expert 数增加一。当前 DeepEP / Mega 系列后端还需要为各个 EP Rank 保留本地 Shared Expert Slot，因此布局会扩展为 `N + EP_size`。Checkpoint Loader 随后把 `mlp.shared_experts` 的权重重映射到追加的 Slot；只改 Expert ID 而不改加载布局，并不能得到正确结果。

## 当前实现的三层控制

![SGLang 共享专家融合的开关、Layout 与调度层](/images/llm-system-sglang-shared-expert-fusion/03-control-layers.png)

当前实现把控制拆成了三层：开关层决定是否产生 Fused Shared Slot，Layout 层决定这些 Slot 在普通 MoE 或 EP 后端中的物理排列，调度层再决定它们由哪个 Rank 执行、是否参与通信计算重叠。只有第一层是在回答“是否融合”；Waterfill 与 SBO / TBO 回答的是另外两个问题。

`--disable-shared-experts-fusion` 用于显式关闭优化；`--enforce-shared-experts-fusion` 用于绕过本应由后端等条件触发的默认关闭，并与前者互斥。强制开关不会自动修复不匹配的 Checkpoint、量化格式或后端语义。

旧版 `determine_num_fused_shared_experts` 中的 DeepSeek-V3 / R1、256 / 384 Experts、单 Shared Expert、Quark 等判断，是特定阶段的保守兼容门禁，不应当作共享专家融合的普遍定义。当前代码路径已经继续演进，例如 GLM 系列也加入了融合适配。

## Fusion、Waterfill 与 SBO / TBO

![Fusion、Waterfill 与 SBO、TBO 的三个优化维度](/images/llm-system-sglang-shared-expert-fusion/04-fusion-waterfill-sbo.png)

Fusion 改的是 Kernel 与 Expert Layout；Waterfill 把已经融合的 Shared Slot 当成一个额外 Routed Slot，并在 EP 场景中发往最轻载 Rank；SBO / TBO 改的是调度，把可独立执行的 Shared Expert 计算放进 A2A 通信空隙。它们的优化对象不同。

早期实现往往在 Fusion 与 SBO / TBO 之间二选一，因为融合后不再有一段独立 Shared Expert GEMM 可供调度。当前实现已经出现面向 EP Waterfill 和 SBO 内融合的进一步路径，但能否组合仍取决于 Backend、版本与具体执行路径，不能只根据某一个开关推断。

## SBO 与 TBO：同批并发 vs 双微批流水

![SGLang SBO 与 TBO 的执行时序对比](/images/llm-system-sglang-shared-expert-fusion/06-sbo-vs-tbo.png)

SBO 不拆分当前 Forward Batch，而是在同一 Batch 内通过 Dispatcher Hook、Stream / Event 与 SM 划分，让 Shared Expert 或 Down GEMM 与 Dispatch / Combine 通信局部并发。TBO 则在 Model Runner 内把一个 Forward Batch 切成两个 Child Micro-Batch，再由两个 Stage Executor 围绕 Yield Point 交替推进，用一边的 Attention / MoE 计算覆盖另一边的 Dispatch / Combine 通信。具体重叠窗口仍会随 Prefill / Decode、Backend 与硬件策略变化。

## Checkpoint 与量化格式

![SGLang 共享专家融合的 Checkpoint 与量化格式约束](/images/llm-system-sglang-shared-expert-fusion/05-checkpoint-remap-contract.png)

当前 Loader 在启用融合时，会把 `mlp.shared_experts` 重映射为 `mlp.experts.N`，再交给 FusedMoE 的 Weight Loader。这个过程看似只是改名，实际要求目标 Slot 能完整接收 Weight、Scale 与 Packed Layout。

Quark MXFP4 是明确的兼容示例：Routed Expert 与 Shared Expert 使用一致的格式和 Scale 语义。对于 Mixed Quant 或独立存储 Shared Expert 的 Checkpoint，不能仅凭名字可重映射就判断兼容，还必须核验 Shape、Dtype、Scale 与实际 Packed Layout。旧版代码中的 384 Experts + 非 Quark、W4AFP8 / W4A16 等限制，反映的是当时这份实现契约尚未满足，而不是所有版本都成立的固定公式。

## 参考实现

- [SGLang `deepseek_v2.py`：融合布局与执行路径](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v2.py)
- [SGLang `deepseek_weight_loader.py`：Shared Expert 权重重映射](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py)
- [SGLang Server Arguments：`--enforce-shared-experts-fusion`](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/server_arguments.mdx)
- [SGLang Expert Parallelism：SBO 与 EP 执行路径](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/expert_parallelism.md)
- [SGLang `single_batch_overlap.py`：SBO 重叠参数](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/batch_overlap/single_batch_overlap.py)
- [SGLang `two_batch_overlap.py`：TBO 拆分与交错执行](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/batch_overlap/two_batch_overlap.py)
- [初始 Shared Expert Fusion 实现 PR #4918](https://github.com/sgl-project/sglang/pull/4918)
- [GLM Shared Expert Fusion 适配 PR #13873](https://github.com/sgl-project/sglang/pull/13873)
