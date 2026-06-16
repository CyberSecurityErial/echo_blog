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

于是本人暂时丢掉了Femtotron的开发，开始参考Megatron-Core的代码实现。pipeline parallel的scheduler个人觉得是训练框架中比较有趣的一个内容，刚好也是在开发这部分，就借此机会把Megatron的核心源码走读一遍。当然可能写的比较草率，后续肯定还会单独整理一个比较完整的博客专门读。这里主要展示读的过程中我脑子里面的cot。

## pipeline parallel的一切
gpt5.5虽然被agenticRL搞得不说人话，但是有一个词用得很好导致我自己也很喜欢，“心智模型”。读各类代码尤其是涉及到并行计算的代码，都需要有这样一个心智模型。比如cuda是simt。编程模型这说法在框架这层就显得有点不够贴切。Megatron的pp代码，心智模型是SPMD，也就是每个Rank进入同一个函数。这个从megaton的使用方式里面也能看出来，启动方式类似于mpi。pp stage里面具体做什么操作，由rank固有的状态决定：包括这个rank属于哪个通信组。Megatron实现vpp的方式就和我预设的完全不一样，它在get_forward_backward_func（）里面先用ifelse判断要不要用1f1b，还是直接简单粗暴的gpipe，在1f1b的ifelse里面又直接用了ifelse判断要不要用vpp，而且给是否开vpp分别写了一个函数，这两个函数分别是forward_backward_pipelining_with_interleaving 和 forward_backward_pipelining_without_interleaving。

Megatron这样实现就可以体现出增量的困难，比如这里就无法设计成“把gpipe改成1f1b”或者把“1f1b增量成interleave 1f1b”的样子。也侧面说明我一开始的设想不适合整体代码架构。

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
写不动了，用ai先混混日子（*）
  从上到下依次执行：

  - 【VPP核心】要求 model 必须是 list[Module]，每个元素是一个 model_chunk；data_iterator 也必须是 list，每个 chunk 一个 iterator。非 VPP 是“不支持 model chunk”，这里正好相反。
  - 初始化 config = get_model_config(model[0])。
  - 初始化 p2p_communicator 和 pg_collection。和无 VPP 类似，p2p_communicator 管 PP send/recv，pg_collection 保存 TP/CP/DP/PP/embedding 等 group。
  - 【区别】VPP 这里没有 multi-module pipeline 的特殊分支，也不支持 adjust_tensor_shapes_fn：

    assert adjust_tensor_shapes_fn is None

  - 如果开启 moe_paged_stash，重置 MoE stash。
  - 【区别】VPP 支持 overlap_p2p_comm，但不能和 batch_p2p_comm 同时开。无 interleave 是直接不支持 overlap_p2p_comm。
  - 如果 M-Core 负责 finalize grads，且不是 forward-only，清空 embedding wgrad deferral buffer。这个和无 VPP 一样，schedule 只清空和最后 drain，中间缓存发生在 layer 里。
  - 禁用梯度同步 disable_grad_sync()。
  - 【VPP区别】no_sync_func 可能是 list，因为每个 model chunk 都可能是一个 DDP/FSDP wrapper。这里用 ExitStack 把所有 chunk 的 no_sync() 一起 enter。
  - 【VPP区别】grad_sync_func 和 param_sync_func 也会被规范成 list，每个 model chunk 一个同步函数。
  - 如果是 forward_only，临时把 grad_sync_func / param_sync_func 置空，结束时再恢复。
  - 初始化 input_tensors / output_tensors / output_tensor_grads。
    【区别】这里是二维结构：

    input_tensors = [[] for _ in range(len(model))]
    每个 model chunk 各自维护一个 FIFO 队列。

  - 检查 microbatch_group_size_per_vp_stage。它必须在 [PP, num_microbatches] 范围内。
  - 【VPP核心】检查最后一个 microbatch group 的 remainder，不能是 0 < remainder < PP，否则会产生 dependency bubble。
  - 计算 tensor_shape = [seq_length, micro_batch_size, hidden_size]，再按 CP / sequence parallel 切分。
  - 【区别】这里没有 recv_tensor_shapes/send_tensor_shapes 两套，也没有 decoder_seq_length 特殊处理；VPP 直接用单个 tensor_shape。
  - 计算：

    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    【VPP核心】后续调度单位是 virtual_microbatch_id，不是普通 microbatch id。

  - 计算 warmup / remaining。VPP warmup 公式本质是：

    (PP - rank - 1) * 2
    + (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
    如果开 overlap_moe_expert_parallel_comm，还会多加 1。

  - 如果配置了 partial activation checkpointing，设置 max_outstanding_backprops = num_warmup_microbatches + 1。
  - 如果配置了参数同步函数，先同步前两个 model chunks 的参数：

    param_sync_func[0](...)
    param_sync_func[1](...)

  ### schedule table

  - 【VPP核心】构造 schedule_table，把 virtual_microbatch_id 映射成：

    (microbatch_id, model_chunk_id)

  - 例如 VP=2 时，不是简单 chunk0, chunk1, chunk0, chunk1 交替，而是受 microbatch_group_size_per_vp_stage 控制，可能先连续跑 chunk0 的 N 个 microbatch，再跑 chunk1 的 N 个。
  - 拆出：

    microbatch_id_table
    model_chunk_id_table

  - get_model_chunk_id(id, forward=True)：forward 按 table 取 chunk。
  - get_model_chunk_id(id, forward=False)：backward 会反转 chunk 顺序：

    model_chunk_id = num_model_chunks - model_chunk_id - 1

  - 【VPP核心】forward chunk 顺序和 backward chunk 顺序是反的。

  ### helper 部分

  - recv_tensor_from_previous_stage(...) 判断当前 virtual microbatch 是否需要收 tensor，以及收到后应该 append 到哪个 model chunk 的队列。
  - 【VPP核心】不是每个 virtual stage 都真的跨 PP rank 通信。
    如果是物理 PP first rank + VP first chunk，forward 不需要 recv。
    如果是物理 PP last rank + VP last chunk，forward 不需要 send。
    backward 方向反过来。

  - forward_step_helper_preprocess(...)：
      - 可能提前触发下一个 model chunk 的 param sync。
      - 如果是第一个 PP rank 的第一个 VP chunk，给该 chunk 的 input 队列补 None。
      - 根据 microbatch_id - offset 从对应 chunk 的 input queue 取 input。

  - forward_step_helper(...)：
      - 根据 virtual id 找到 model_chunk_id 和 microbatch_id。
      - 调 forward_step(...)，传入 model[model_chunk_id] 和 data_iterator[model_chunk_id]。
      - 【区别】只有 VP last stage + PP last stage 才算真正 last stage / loss。
      - forward 后把 output append 到 output_tensors[model_chunk_id]，累计 total_num_tokens。

  - backward_step_helper(...)：
      - 根据 virtual id 反向找到 backward 的 chunk。
      - 从对应 chunk 的 input_tensors/output_tensors/output_tensor_grads 队列 pop。
      - 调 backward_step(...) 得到 input_tensor_grad。

  - 【VPP梯度同步区别】默认同步时，如果这是某个 model chunk 的最后一个 microbatch，会在该 chunk 的 backward 前 enable_grad_sync()；如果有 grad_sync_func，则在 backward 后按 chunk 显式调用对应的 grad_sync_func[chunk_id]。
  - forward_backward_helper_wrapper(...)：
      - 普通情况：分别调 forward helper 和 backward helper。
      - 【区别】如果开启 overlap_moe_expert_parallel_comm，会走 combined_1f1b_schedule_for_interleaved_pipelining，把 MoE expert comm overlap 也揉进 1F1B。

  ### warmup 阶段

  - 一开始先做一次：

    input_tensors[0].append(recv_forward(...))
    【区别】无 VPP 是 warmup loop 里每轮 recv_forward；VPP 先给 chunk0 准备第一个 input。

  - 初始化异步通信相关的 wait handles 和 recv buffers。
  - 【区别】如果开 overlap_p2p_comm_warmup_flush，warmup/cooldown 也会用异步 prefetch buffer。
  - 对 k in range(num_warmup_microbatches)：
      - 根据 k 找当前 forward 的 cur_model_chunk_id。
      - 如果 warmup flush overlap 开启，先 wait 上一次预取的 forward recv handle。
      - 调 recv_tensor_from_previous_stage(k, forward=True)，判断是否需要接收下一个 forward input，以及它属于哪个 chunk。
      - 计算 checkpoint_activations_microbatch，逻辑和无 VPP 类似。
      - 调 forward_backward_helper_wrapper(f_virtual_microbatch_id=k) 做 forward。
      - 如果当前是 VP last + PP last，output_tensor = None，因为没有下游。
      - 如果不开 warmup flush overlap：
          - 普通轮次走 send_forward_recv_forward，发送当前 output，同时接收下一轮 forward input。
          - 【边界】warmup 最后一轮、训练模式、且后面还有 steady 1F1B 时，可能走 send_forward_backward_recv_forward_backward，同时接第一个 backward grad。

      - 如果开 warmup flush overlap：
          - 使用 async send_forward_recv_forward(..., overlap_p2p_comm=True)。
          - wait send handle 后才能 deallocate output，避免异步 send 还没拷完就释放源 buffer。

      - 收到的 forward input append 到 input_tensors[next_forward_model_chunk_id]。
      - deallocate_output_tensor(output_tensor, ...) 省显存。
      - 如果开 overlap_p2p_comm，warmup 最后一轮还会提前发起 backward recv，把收到的 grad 放到 output_tensor_grads[num_model_chunks - 1]。

  ### steady 1F1B 阶段

  - 对 k in range(num_microbatches_remaining)：

    forward_k = k + num_warmup_microbatches
    backward_k = k

  - 【VPP核心】forward 和 backward 都用 virtual id。forward chunk 按正序，backward chunk 通过 get_model_chunk_id(..., forward=False) 反序。
  - 计算当前 forward 的 checkpoint/recompute 配置。
  - 如果 config.overlap_p2p_comm=True：
      - 定义 pp_pre_forward：等 forward recv handle，确保 input 到了。
      - 定义 pp_post_forward：异步 send 当前 output，同时异步 recv 下一个 forward input，并 append 到对应 chunk。
      - 定义 pp_pre_backward：等 backward recv handle，确保 output grad 到了。
      - 定义 pp_post_backward：异步 send input grad，同时异步 recv 下一个 output grad，并 append 到对应 chunk。
      - 调 wrapper，同时完成一个 forward virtual microbatch 和一个 backward virtual microbatch。

  - 如果不开 p2p overlap：
      - 先 wrapper 做：

        forward(forward_k) + backward(backward_k)

      - 如果当前 forward 是 VP last + PP last，不 send forward output。
      - 如果当前 backward 是 VP first + PP first，不 send backward input grad。
      - 计算 recv_prev/recv_next 和对应 chunk。
      - 调：

        send_forward_backward_recv_forward_backward(...)
        同时完成：

        send forward activation
        send backward input grad
        recv next forward activation
        recv next backward output grad

      - 收到的 forward input append 到对应 input_tensors[chunk]。
      - 收到的 backward grad append 到对应 output_tensor_grads[chunk]。
      - deallocate 当前 output。

  - 【区别】无 VPP 的 steady 是一个全局 FIFO；VPP 是每个 chunk 各自 FIFO，并且通信收到的 tensor 要按 next_forward_model_chunk_id / next_backward_model_chunk_id 放回正确队列。

  ### cooldown 阶段

  - 只在训练模式执行。
  - 先 wait 还没完成的 backward async handles。
  - 如果所有 microbatch 都在 warmup，没有进入 steady，则先 recv_backward 拿第一个 backward grad，放到最后一个 chunk 的 output_tensor_grads。
  - 对：

    k in range(num_microbatches_remaining, total_num_microbatches)
    执行剩余 backward。

  - 根据 get_model_chunk_id(k, forward=False) 找当前 backward chunk。
  - 如果需要，先 wait backward recv handle。
  - recv_tensor_from_previous_stage(k, forward=False) 判断是否还要从下游收 grad，以及 grad 应该放到哪个 chunk。
  - 如果开 warmup flush overlap，可能先 async prefetch 下一个 backward grad。
  - 调：

    forward_backward_helper_wrapper(b_virtual_microbatch_id=k)
    这里只做 backward，不做 forward。

  - 如果当前是 VP first + PP first，input_tensor_grad = None，因为没有上游。
  - 然后发送 backward grad，并可能接收下一个 output grad：
      - overlap flush：走 async send_backward_recv_backward
      - 普通路径：走同步 send_backward_recv_backward

  - 收到的 output grad append 到对应 chunk 的 output_tensor_grads。
  - cooldown 结束后，wait send_prev_wait_handle。
  - 【VPP梯度同步区别】最后会：

    enable_grad_sync()
    然后对还没同步过的 model chunks 调 grad_sync_func[chunk_id](...)。
    它用 synchronized_model_chunks 避免一个 chunk 重复同步。

  ### misc / finalize 阶段

  - 检查所有 async recv wait handles 都清空：

    assert not recv_prev_wait_handles
    assert not recv_next_wait_handles

  - 如果训练模式且 M-Core 负责 finalize：
      - finish_embedding_wgrad_compute(...)：把 embedding/output layer 延迟的 wgrad GEMM 补算完。
      - finalize_model_grads_func(...)：做 DP 梯度 all-reduce/reduce-scatter、SP layernorm grad 同步、PP embedding grad 同步、global token 缩放。

  - 【区别】这里传给 finalize 的是 model 本身，因为 VPP 下 model 已经是 chunk list；无 VPP 里要传 [model]。
  - 如果开 fine-grained activation offloading，reset offload interface。
  - 如果 forward-only，恢复之前临时置空的 grad_sync_func/param_sync_func。
  - stop timer，必要时创建 cudagraph，返回 forward_data_store。

  ### 最关键的 VPP 区分点

  - 【调度单位不同】无 VPP 用 microbatch_id；VPP 用 virtual_microbatch_id，再映射到 (microbatch_id, model_chunk_id)。
  - 【队列结构不同】无 VPP 是单个 FIFO；VPP 是每个 model chunk 一个 FIFO。
  - 【forward/backward chunk 顺序不同】forward 按 chunk 正序，backward 按 chunk 反序。
  - 【通信落点不同】收到的 activation / grad 不能直接放全局队列，要根据 next_forward_model_chunk_id / next_backward_model_chunk_id 放到对应 chunk。
  - 【梯度同步粒度不同】无 VPP 基本按整个 model；VPP 要按 model chunk 跟踪同步状态。
  - 【P2P overlap 支持不同】无 VPP 不支持 overlap_p2p_comm；VPP 支持，并且额外有 warmup/flush overlap 分支。
  - 【shape 处理不同】无 VPP 有 recv_tensor_shapes/send_tensor_shapes/adjust_tensor_shapes_fn；VPP 这里只用一个 tensor_shape，且不支持 adjust_tensor_shapes_fn。
