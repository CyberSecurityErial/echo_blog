---
date: '2026-06-24T00:00:00+08:00'
draft: true
title: 'LLM System: 深入低精度量化 00'
categories: ["LLM System"]
tags: ["LLM", "LLM System", "Quantization", "Low Precision"]
series: ["LLM System", "深入低精度量化"]
series_order: 0
weight: 1
math: true
---

## 说明
不必要的文字交给AI

## 查表

![低精度格式 bit 和 scale](../../images/llm-system-low-precision-quantization/precision-bit-layout.svg)

## 为什么量化好？
省显存
### 为什么有tf32？
### 为什么有fp16还要bf16？
### 为什么没有bf8和bf4？
### 为什么fp量化比int量化好？
int是最小和最大之间均匀标尺。fp是任意组合科学计数法只要不超过位数。
fp这种动态表示相比于int牺牲了点什么？牺牲了局部的密度，刻画普通值的能力变稀疏了。int的量化间距是const s，fp的间距是2^e
### 为什么fp4量化这么nb？
### 什么时候用细粒度量化？
统计学问题，要看有没有很大的异常值。另外低精度更需要细粒度量化，因为能表示的范围更小，就更容易出现被一个高值拖累整体的现象。
### int真的总不如fp吗

拿 4-bit 举例。INT 是先 round 到整数码本，再乘回 scale：

{{< math >}}
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{X}{s}\right), -Q_{\max}, Q_{\max}\right),
\qquad
X_q = q \cdot s
{{< /math >}}

FP 是先算 $X/s$，再找最近的低比特 FP 码本值：

{{< math >}}
q = \operatorname{nearest}_{\mathcal{C}_{fp}}\left(\frac{X}{s}\right),
\qquad
X_q = q \cdot s
{{< /math >}}

scale 通常取：

{{< math >}}
s = \frac{\operatorname{AbsMax}(X)}{Q_{\max}}
{{< /math >}}

这里看两个码本：

```text
MXINT4: Qmax = 7, codebook = {-7, ..., 0, ..., 7}
MXFP4:  Qmax = 6, codebook+ = {0, 0.5, 1, 1.5, 2, 3, 4, 6}
```

### 例子 1：有 outlier

{{< math >}}
X = [1,\ 2,\ 4,\ 12]
{{< /math >}}

INT4：

{{< math >}}
s = \frac{12}{7} = 1.714
{{< /math >}}

| 原值 $x$ | $x/s$ | round 后 $q$ | 反量化 $x_q$ |
| ---: | ---: | ---: | ---: |
| 1 | 0.583 | 1 | 1.714 |
| 2 | 1.167 | 1 | 1.714 |
| 4 | 2.333 | 2 | 3.429 |
| 12 | 7.000 | 7 | 12.000 |

{{< math >}}
X_q^{int4} = [1.714,\ 1.714,\ 3.429,\ 12]
{{< /math >}}

FP4：

{{< math >}}
s = \frac{12}{6} = 2,
\qquad
\frac{X}{s} = [0.5,\ 1,\ 2,\ 6]
{{< /math >}}

| 原值 $x$ | $x/s$ | 最近 FP4 码 | 反量化 $x_q$ |
| ---: | ---: | ---: | ---: |
| 1 | 0.5 | 0.5 | 1 |
| 2 | 1 | 1 | 2 |
| 4 | 2 | 2 | 4 |
| 12 | 6 | 6 | 12 |

{{< math >}}
X_q^{fp4} = [1,\ 2,\ 4,\ 12]
{{< /math >}}

### 例子 2：没有大 outlier

{{< math >}}
X = [2.5,\ 3.0,\ 3.5,\ 4.0]
{{< /math >}}

INT4：

{{< math >}}
s = \frac{4}{7} = 0.571
{{< /math >}}

| 原值 $x$ | $x/s$ | round 后 $q$ | 反量化 $x_q$ |
| ---: | ---: | ---: | ---: |
| 2.5 | 4.375 | 4 | 2.286 |
| 3.0 | 5.250 | 5 | 2.857 |
| 3.5 | 6.125 | 6 | 3.429 |
| 4.0 | 7.000 | 7 | 4.000 |

{{< math >}}
X_q^{int4} = [2.286,\ 2.857,\ 3.429,\ 4.000]
{{< /math >}}

FP4：

{{< math >}}
s = \frac{4}{6} = 0.667,
\qquad
\frac{X}{s} = [3.75,\ 4.5,\ 5.25,\ 6]
{{< /math >}}

| 原值 $x$ | $x/s$ | 最近 FP4 码 | 反量化 $x_q$ |
| ---: | ---: | ---: | ---: |
| 2.5 | 3.75 | 4 | 2.667 |
| 3.0 | 4.5 | 4 | 2.667 |
| 3.5 | 5.25 | 6 | 4.000 |
| 4.0 | 6.0 | 6 | 4.000 |

{{< math >}}
X_q^{fp4} = [2.667,\ 2.667,\ 4.000,\ 4.000]
{{< /math >}}

所以：

```text
FP:  动态范围大，遇到 outlier 更舒服
INT: 局部范围小的时候，均匀格子更细
```

## PTX 指令

### Ampere / sm_80

#### cvt

```ptx
cvt.satfinite.{f16,bf16,f16x2,bf16x2,tf32}.f32
cvt.f32.bf16
```

#### mma / TF32 / BF16

```ptx
mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 d, a, b, c;

mma.sync.aligned.m16n8k8.row.col.f32.atype.btype.f32 d, a, b, c;
.atype = {.bf16, .tf32};
.btype = {.bf16, .tf32};

mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 d, a, b, c;
```

#### mma / INT8 / INT4

```ptx
mma.sync.aligned.shape.row.col{.satfinite}.s32.atype.btype.s32 d, a, b, c;
.atype = {.u8, .s8};
.btype = {.u8, .s8};

mma.sync.aligned.shape.row.col{.satfinite}.s32.atype.btype.s32 d, a, b, c;
.atype = {.u4, .s4};
.btype = {.u4, .s4};
```

### Hopper / sm_90

#### cvt / FP8

```ptx
cvt.satfinite.{e4m3x2,e5m2x2}.{f32,f16x2}
```

#### wgmma / FP8

```ptx
wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e5m2
wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e4m3
```

#### wgmma / INT8

```ptx
wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite
wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8
wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8.satfinite
wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8
```

#### wgmma / sync

```ptx
wgmma.fence.sync.aligned;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

### Blackwell / sm_100 / sm_120

#### cvt / FP4 / FP6 / FP8

```ptx
cvt.rn.satfinite{.relu}.{e2m1x2,e2m3x2,e3m2x2,ue8m0x2}.f32
cvt.rn{.relu}.f16x2.{e2m1x2,e2m3x2,e3m2x2}
cvt.rs{.e2m1x4,.e4m3x4,.e5m2x4,.e3m2x4,.e2m3x4}.f32
cvt.rn.satfinite{.relu}{.e2m3x2,.e3m2x2,.e2m1x2}.{f16x2,bf16x2}
cvt.rn{.satfinite}{.relu}.bf16x2.{e4m3x2,e5m2x2,e3m2x2,e2m3x2,e2m1x2}
```

#### mma / FP8 / FP6 / FP4

```ptx
mma.sync.aligned.m16n8k32.row.col.kind.dtype.f8f6f4type.f8f6f4type.ctype d, a, b, c;

.kind = {.kind::f8f6f4};
.f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};
.ctype = {.f16, .f32};
.dtype = {.f16, .f32};
```

#### mma / block_scale / MXFP4

```ptx
mma.sync.aligned.m16n8k64.row.col.kind.block_scale{.scale_vec_size}.f32.e2m1.e2m1.f32.stype
d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};

.kind = {.kind::mxf4};
.scale_vec_size = {.scale_vec::2X};
.stype = {.ue8m0};
```

#### mma / block_scale / MXFP8-FP6-FP4

```ptx
mma.sync.aligned.m16n8k32.row.col.kind.block_scale{.scale_vec_size}.f32.f8f6f4type.f8f6f4type.f32.stype
d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};

.kind = {.kind::mxf8f6f4};
.scale_vec_size = {.scale_vec::1X};
.f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};
.stype = {.ue8m0};
```

#### tcgen05.mma

```ptx
tcgen05.mma.cta_group.kind [d-tmem], a-desc, b-desc, idesc,
{ disable-output-lane }, enable-input-d {, scale-input-d};

.kind = {.kind::f16, .kind::tf32, .kind::f8f6f4};
.cta_group = {.cta_group::1, .cta_group::2};
```

#### tcgen05.mma / block_scale

```ptx
tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
[d-tmem], a-desc, b-desc, idesc,
[scale-A-tmem], [scale-B-tmem], enable-input-d;

.kind = {.kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4};
.scale_vectorsize = {.scale_vec::1X, .scale_vec::2X, .scale_vec::4X, .block16, .block32};
```

#### tcgen05.mma.sp / block_scale

```ptx
tcgen05.mma.sp.cta_group.kind.block_scale{.scale_vectorsize}
[d-tmem], a-desc, b-desc, [sp-meta-tmem], idesc,
[scale-A-tmem], [scale-B-tmem], enable-input-d;

.kind = {.kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4};
```

### 最简表

| 代际 | PTX |
| --- | --- |
| Ampere / sm_80 | `cvt.satfinite.{f16,bf16,f16x2,bf16x2,tf32}.f32`; `cvt.f32.bf16`; `mma.sync...tf32`; `mma.sync...bf16`; `mma.sync...s32.u8/s8`; `mma.sync...s32.u4/s4` |
| Hopper / sm_90 | `cvt...e4m3x2/e5m2x2`; `wgmma.mma_async...e4m3/e5m2`; `wgmma.mma_async...s8/u8`; `wgmma.fence/commit_group/wait_group` |
| Blackwell / sm_100/sm_120 | `cvt...e2m1/e2m3/e3m2/ue8m0`; `cvt.rs...x4`; `mma.sync...kind::f8f6f4`; `mma.sync...block_scale...kind::mxf4/mxf8f6f4`; `tcgen05.mma...kind::mxf*` |

### 思考量化操作的代际演化
为什么a到h支持mma的ab矩阵不同量化精度：wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e4m3。量化精度是耦合在矩阵本身语义的，一种数据分布有自己的敏感度和适应度，也就有自己的最好的量化格式。那么是否可以设计实验验证呢？TODO。另外还有个优点指令少一条，可以类似于一种隐式强制cvt。

h到b进化了什么。单线程可以发动tcgen05.mma无需collective。A可以在tmem和smem，B在smem，C在tmem。对矩阵的属性用desc描述。wgmma和tcgen05.mma都有同步，但是tcgen05强调的是kernel要求计算这个kernel的所有cta级同步。深入点说wgmma更类似于超长指令字，每个warp是一个指令。而tcgen05是类似于片上资源（tmem）布局的声明，有点像GPU写了一个kernel desc，把这个desc作为plan丢给tensorCore让tcore自己再LaunchKernel（逃）。


