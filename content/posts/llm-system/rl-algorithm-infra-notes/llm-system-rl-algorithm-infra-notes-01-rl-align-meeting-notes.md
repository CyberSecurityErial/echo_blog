---
date: '2026-06-21T00:00:00+08:00'
draft: false
title: 'LLM System: 算法和 Infra 交织的 RL 杂谈 01 - RL Align 会议纪要与一点思考（AI 总结）'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "RL", "RLHF", "Alignment", "AI Infra", "Meeting Notes"]
series: ["LLM System", "算法和 Infra 交织的 RL 杂谈"]
series_order: 1
weight: 1
math: true
---

以下是完整精简版，已去掉群聊表述，并把 AFD 按 **AF 分离** 重新纳入主线。

# RL AIGC 开发者交流纪要：从模型适配到异步训练系统

这次交流的核心不是单一算法，而是 RL AIGC 训练在工程落地中的系统问题。整体看下来，主要矛盾是：RL 链路把训练、推理、数据流、权重同步、checkpoint 和调试工具全部耦合在一起，而现有框架对这些问题的支持还不够完整。

---

## 1. 多模态 RL 的更新稳定性

多模态 RL 中，有一种做法是：如果某次参数更新和当前模型之间的 diff 超过阈值，就直接舍弃这次更新。

这个机制可以避免异常 update 破坏模型状态：

```text
异常 batch / 异常 reward / 异常 rollout
        ↓
参数更新过大
        ↓
超过阈值后舍弃本次更新
```

但它只能止损，不能解释问题来源。真正需要的是面向 RL 的 debugger，能够定位是 reward、logprob、rollout、并行切分还是权重同步出了问题。

---

## 2. 新模型接入成本高

如果要把一个新模型接入 RL 训练框架，往往需要手写 Megatron、FSDP 或其他并行逻辑的适配。

难点不只是 forward 能跑，而是整个 RL 链路都要对齐：

```text
模型结构
并行切分
checkpoint / reshard
rollout 权重同步
logprob 计算
loss 计算
训练侧和推理侧的数据格式
```

RL 场景下，模型适配错误不一定马上报错，很多时候只表现为训练逐渐崩掉。因此新模型适配需要更强的调试工具，比如检查权重版本、logprob 对齐、reshard 正确性和并行切分一致性。

---

## 3. RL 训练周期长，问题复现成本高

RL 训练周期通常很长，一个周期可能需要几天。很多问题不会在前几个 step 暴露，而是在训练一段时间后才出现。

这会导致调试非常低效：

```text
训练时间长
    ↓
问题出现晚
    ↓
复现实验贵
    ↓
难以判断是算法问题还是系统问题
```

所以 RL 系统不能只看最终 reward 或 loss，还需要记录中间状态，例如 reward 分布、logprob 分布、policy version、rollout 质量和异常 batch。

---

## 4. verl + omni 生态仍需完善

verl 和 omni 相关方向已经具备基本框架，但生态还不够成熟，很多能力仍然需要补齐。

主要缺口包括：

```text
多模型适配
多模态支持
调试工具
checkpoint / resume
权重同步
推理引擎对接
大规模训练验证
```

RL 框架不是只实现 PPO、GRPO 就够了。真正可用的框架还需要解决模型、数据、推理、并行和调试之间的系统问题。

---

## 5. 小算力场景需要 LoRA / QLoRA 支持

对于算力较少的团队，全参 RL 或大规模全参微调并不现实，更多时候只能做 LoRA 或 QLoRA。

因此 RL 框架需要支持低成本训练路径：

```text
LoRA 参数如何参与训练
LoRA 参数如何同步到 rollout 侧
LoRA checkpoint 如何保存
QLoRA 下量化权重和 adapter 如何配合
FSDP / TP / PP 场景下 adapter 如何切分
```

如果框架只优先支持全参训练，会限制中小规模团队使用。

---

## 6. AFD：AF 分离的价值和难点

这里的 AFD 可以理解为 **A/F Decoupling**，即 **AF 分离**。

```text
A = Actor / 训练侧
F = Forward / Rollout / 推理侧
```

AF 分离的目标是把训练侧和推理侧拆开：

```text
Actor 训练侧：
负责 backward、optimizer step、权重更新

Forward / Rollout 推理侧：
负责 generation、sampling、logprob、环境交互
```

这个方向本身是合理的。训练和推理的系统需求不同：

```text
训练侧关心：
梯度、优化器、FSDP、ZeRO、Megatron、checkpoint

推理侧关心：
batching、KV cache、decoding、低延迟、高吞吐
```

把二者拆开，可以让训练系统和推理系统各自优化。

但 AF 分离真正难的地方在于：A 侧和 F 侧之间不是简单的一对一关系，而是动态 M:N 关系。

例如：

```text
M 个 Actor training ranks
        ↓
N 个 Forward / Rollout workers
```

训练侧和推理侧的并行拓扑可能完全不同。训练侧可能使用 TP、PP、FSDP、ZeRO，推理侧可能使用另一套 tensor parallel、batching 和 KV cache 管理方式。

因此系统必须解决：

```text
训练权重如何同步到推理侧
训练 layout 如何 reshard 成推理 layout
多个 actor rank 如何对应多个 rollout worker
rollout 数据应该送回哪个 trainer
policy version 如何管理
stale rollout 是否允许
```

如果动态 M:N 处理不好，AF 分离就会显得很尴尬：理论上拆开更优雅，实践中却会被权重同步、reshard、数据回流和版本管理拖住。

---

## 7. rollout pool 不是普通 queue

全异步 RL 需要把 rollout 结果放入一个 pool，再由训练侧消费。

简单结构是：

```text
rollout worker
      ↓
rollout pool
      ↓
trainer
```

但这个 pool 不能只是 Python queue。它至少需要处理：

```text
多 producer / 多 consumer
异步读写
backpressure
数据新鲜度
样本重复和丢失
policy version
checkpoint / resume
```

更准确地说，rollout pool 应该是一个带版本管理的数据池。

每条 rollout 数据都需要知道：

```text
来自哪个 policy version
是否已经被消费
是否已经过期
应该被哪个 trainer 消费
resume 后是否还能继续使用
```

这也是全异步 RL 的核心系统问题之一。

---

## 8. rollout pool 和 dataloader 的相似性

rollout pool 和 dataloader 有相似结构。

dataloader 是：

```text
多个 dataloader worker
        ↓
prefetch queue / buffer
        ↓
training process
```

rollout pool 是：

```text
多个 rollout / inference worker
        ↓
rollout pool
        ↓
trainer
```

二者都是多 worker 异步生产数据，再由训练侧消费。

因此它们都会遇到 worker 级别的问题：

```text
worker 如何划分数据
worker 之间是否会重复
worker 输出如何路由
worker 失败如何恢复
checkpoint 后从哪里继续
```

分布式训练只解决 rank 级别的问题，不会自动解决 worker 级别的数据划分。这里的关键不是 queue 本身，而是 worker-aware 的异步数据系统。

---

## 9. NCCL、reshard 和节点内外通信

AF 分离之后，训练侧的权重或状态需要送到 rollout 侧。由于训练和推理的并行 layout 不同，通常需要 reshard。

理想路径是：

```text
Actor 训练侧权重
        ↓
reshard 一次
        ↓
rollout 推理侧使用
        ↓
同节点内尽量走 CUDA IPC 等高速路径
        ↓
跨节点通信和节点内搬运尽量 overlap
```

这个方向理论上可以减少通信开销，但实践中不容易做好。

难点在于：

```text
训练侧和推理侧拓扑不同
rollout 节奏不固定
权重版本持续变化
reshard 成本高
跨节点和节点内通信需要协调
```

因此这类优化不能只看通信算子本身，还要和权重版本、rollout 调度、训练 step 绑定起来看。

---

## 10. 全异步 RL 的权重同步还不成熟

全异步 RL 中，训练侧不断更新权重，rollout 侧不断使用某个版本的权重生成数据。

这会出现版本差：

```text
trainer 已更新到 step 100
rollout worker 仍在使用 step 95 的权重
```

同步太频繁，通信成本高；同步太少，rollout 数据变旧。

因此系统需要在两者之间权衡：

```text
更高吞吐
vs
更高数据新鲜度
```

目前一些方案实现了局部同步，但主流框架对全异步权重同步的支持仍然有限。

---

## 11. 训推一致性：复用推理侧 logprob

一种保证训推一致性的办法是：rollout 侧生成 response 时，同时计算 logprob，并把这个 logprob 直接交给训练侧使用。

也就是：

```text
rollout 侧生成 response
        ↓
rollout 侧计算 logprob
        ↓
trainer 直接使用该 logprob
```

这样可以减少训练侧和推理侧重复计算 logprob 带来的不一致。

不一致可能来自：

```text
tokenization
padding / mask
position id
attention mask
权重版本
并行切分
kernel 数值差异
```

直接复用 rollout 侧 logprob，可以把一部分一致性问题前移到推理侧。

但它也要求系统记录清楚：

```text
logprob 对应哪个 policy version
logprob 是否和 response 对齐
这条数据是否已经过期
训练侧是否需要重新校验
```

---

## 12. logprob 漂移可能是系统 bug

rollout 和 train 算出来的 logprob 可能随着 step 增长出现偏差。这个问题可能来自数值误差，也可能来自框架实现错误。

常见原因包括：

```text
policy version 对不上
old_logprob 和 new_logprob 混用
mask 错误
position id 错误
reshard 后参数错位
FSDP / TP 切分恢复错误
checkpoint resume 后状态不一致
```

这类问题很危险，因为它不一定立刻报错，而是表现为训练效果逐渐变差。

因此 RL 框架需要专门的 logprob 对齐测试：

```text
同一批 prompt
同一批 response
同一份权重
rollout 侧计算 logprob
train 侧计算 logprob
逐 token 对齐
逐 rank 对齐
resume 后再次对齐
```

这样才能区分算法不稳定和系统实现错误。

---

## 总结

RL AIGC 训练的核心挑战已经不只是算法实现，而是完整系统工程。

关键问题包括：

```text
模型适配
AF 分离
动态 M:N 调度
rollout pool
权重同步
reshard
checkpoint
logprob 对齐
训推一致性
```

AF 分离是一个合理方向：训练侧和推理侧应该各自优化。但 AF 分离之后，系统必须处理动态 M:N、权重同步、reshard、rollout 数据池和 policy version 管理。

如果这些问题没有解决，AF 分离会在工程上变得很别扭。

一句话总结：

> RL 训练框架未来的核心能力，不只是实现 PPO 或 GRPO，而是把训练、推理、数据池、权重同步、checkpoint 和调试工具整合成一个可靠的异步分布式系统。
