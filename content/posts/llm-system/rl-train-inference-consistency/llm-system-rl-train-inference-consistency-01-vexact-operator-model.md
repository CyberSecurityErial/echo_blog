---
date: '2026-06-27T00:00:00+08:00'
draft: true
title: 'LLM System: RL 训推一致 01 - VeRL-VeXact 算子/模型层训推一致处理'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "RL", "RLHF", "Train-Inference Consistency", "verl", "VeXact", "AI Infra"]
series: ["LLM System", "RL 训推一致"]
series_order: 1
weight: 1
math: true
---

# LLM System: RL 训推一致 01 - VeRL-VeXact 算子/模型层训推一致处理

## fsdp初始化 batch-invar op
初始化阶段的算子位于vexact.batch_invariant_ops，主要保证batch invar。
（TODO：这里留个问题，verl只能默认fsdp？为何？）
Qwen3-MoE & DeepSeek-V3 model align

Qwen3适配的算子：

| 需要对齐的算子/模块 | 说明 | 位置 |
| --- | --- | --- |
| MoE block forward | 替换 Qwen3MoeSparseMoeBlock.forward，避免 HF stock MoE 和 VeOmni actor MoE 路径不同 | `vexact/vexact/models/qwen3_moe/modeling_qwen3_moe.py:291` |
| Expert 权重 layout | 使用 fused gate_up_proj + down_proj，对齐 VeOmni v5 actor 侧 fused expert layout | `vexact/vexact/models/qwen3_moe/modeling_qwen3_moe.py:42` |
| Expert MLP 计算 | 走 veomni.ops.fused_moe_forward，确保 rollout 和 actor 的 expert GEMM/dispatch/gather 算法一致 | `vexact/vexact/models/qwen3_moe/modeling_qwen3_moe.py:86` |
| Router softmax/topk/topk norm | 在 patched moe_block_forward 里执行，避免 router 后处理路径和 actor 不一致 | `vexact/vexact/models/qwen3_moe/modeling_qwen3_moe.py:119` |
| MoE 权重加载 | checkpoint separate expert weights 和 fused runtime weights 对齐 | `vexact/vexact/models/qwen3_moe/modeling_qwen3_moe.py:193` |
| 可选 Liger RoPE/SwiGLU | 只有 VEXACT_LIGER_PATCH=1 时启用；注意 LigerRMSNorm 注释明确没启用，因为不是 batch-invariant | `vexact/vexact/models/register.py:69` |

所以 Qwen3-MoE 的重点是：MoE router + topk + expert dispatch/gather + expert GEMM + fused weight layout。


DeepSeek-V3需要对齐的算子是：
 DeepSeek-V3
  DeepSeek-V3 对齐项更多，因为它有 MLA attention、MoE、RoPE、RMSNorm：

| 需要对齐的算子/模块 | 说明 | 位置 |
| --- | --- | --- |
| MoE router | 替换为 VeOmni 对齐的 PatchDeepseekV3TopkRouter，router linear 用 fp32 | `vexact/vexact/models/deepseek_v3/modeling_deepseek_v3.py:86` |
| MoE expert MLP | 替换 DeepseekV3MoE，使用 fused gate_up_proj/down_proj layout | `vexact/vexact/models/deepseek_v3/modeling_deepseek_v3.py:104` |
| Fused MoE kernel | fused path 走 veomni.ops.fused_moe_forward，对齐 actor/rollout expert 算法 | `vexact/vexact/models/deepseek_v3/modeling_deepseek_v3.py:144` |
| Attention / MLA | 替换 DeepseekV3Attention.forward，避免 actor 侧 stock attention padding V 到 qk_head_dim 后走不同 FA codegen | `vexact/vexact/models/deepseek_v3/modeling_deepseek_v3.py:225` |
| RoPE | 替换 DeepseekV3RotaryEmbedding.forward，用 deterministic triton_bmm 计算 freqs | `vexact/vexact/models/deepseek_v3/modeling_deepseek_v3.py:409` |
| RMSNorm | 替换为 batch_invariant_rms_norm，actor 和 rollout 都 patch | `vexact/vexact/models/deepseek_v3/modeling_deepseek_v3.py:487` |
| 权重加载/layout | fused experts 权重名/layout 和 checkpoint/actor FSDP sync 对齐 | `vexact/vexact/models/deepseek_v3/modeling_deepseek_v3.py:323` |

  所以 DeepSeek-V3 的重点是：MoE、MLA attention、RoPE、RMSNorm、fused expert weight layout。

  一句话概括：

  - Qwen3 dense：主要靠通用 batch-invariant GEMM/attention/logprob，对模型本身 patch 很少。
  - Qwen3-MoE：重点对齐 MoE router/expert/fused weight layout。
  - DeepSeek-V3：重点对齐 MoE + MLA attention + RoPE + RMSNorm。

## 模型同步
1. 统一用HFmodel定义（对齐模型结构，modulename，forward逻辑，ckpt格式）
2. 模型patch：apply_veomni_fused_moe_patch + fsdp & rollout weight layout align

## 禁用fa v3
在框架里面叫kvsplit

## cudaGraph
因为训推算子对齐，所以可以用graph捕获算子。

## 小结
到这里也可以大概给出一些总结，训推一致是一个需要多层次解决的问题，算法层做裁剪，框架层做调度，算子层做对齐。
所以这篇blog才单独把VeXact单独拿出来讲，因为VeXact属于算子层面的对齐。接下来的一篇会讲讲VeRL框架这一层需要分管哪一层次的对齐。（当然，这不意味着其他RL框架也只处理这一层级的对齐）
