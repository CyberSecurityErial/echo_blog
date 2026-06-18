---
date: '2026-06-16T00:00:00+08:00'
draft: false
title: 'LLM Theory 杂谈：我也不知道主题是什么，一些零碎的insight'
categories: ["LLM Theory"]
tags: ["LLM", "LLM Theory", "Probability", "Weight", "Optimization"]
series: ["LLM Theory", "杂谈"]
series_order: 1
weight: 1
math: true
---

## softmax

softmax的好处：平滑，稳定，和为1可以保证output的范数有界。如果和小于1就会让norm趋向0，因为自迭代系统会跌入局部吸引子（一个动力系统的insight）。softmax本身就是sigmod的高维版本。soft本身就是改良sig在传统多分类的问题，然后导出的向量级别sig，本质上就是sig pro，sigmod本身假设多标签分类结果非唯一。这就是标量化的坏处啊，看不到上下文，所以要向量化。向量化看不到上下文就要矩阵化，矩阵化看不到上下文就要张量化，然后并行计算就又要退回到向量化，那么就要用向量化逼近矩阵化。

模型表达能力在不断提高上下文阶数；系统实现又必须把高阶上下文拆成可并行的低阶块。优秀算法的本质，就是在这两者之间做近似、重排、分块和融合。

为什么一开始要叫softmax，它一开始被命名成这个是因为人们发现它可以让任何一个数稍微大一点就几乎接近1，所谓softmax，就是可微的max。它的数学性质是，把差值相等转化成比例相等。

具体说说为什么soft是sig的扩展，因为sig是二分类，soft生成一个多分类的概率条，让他们自己进行博弈。
