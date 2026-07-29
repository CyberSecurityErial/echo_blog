---
date: '2026-06-25T00:00:00+08:00'
draft: true
title: '搜广推 Infra & 算法 01 - 搜广推入门'
categories: ["搜广推"]
tags: ["搜索", "广告", "推荐", "Infra", "算法"]
series: ["搜广推 Infra & 算法"]
series_order: 1
weight: 1
math: true
---

## 推荐流程
召回Recall--粗排Pre-Rank--精排Rank--重排ReRank--记录用户日志--离线更新推荐模型
如何保证用户个性化？用户的特征也是一种输入。

## 搜索流程
离线建索引（name major转化成feature major）
查询Query--查询优化（分词，错别字，扩展意思）--召回Retrivial--合并（去掉用户屏蔽的商品，无货商品等）--粗排--精排--重排--展示（好看的前端）--日志--更新离线模型

## 广告流程