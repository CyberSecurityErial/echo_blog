---
date: '2026-05-10T10:48:50+08:00'
draft: false
title: '环境踩坑：wsl+proxy+codex cli'
categories: ["工程踩坑"]
tags: []
series: ["工程踩坑"]
series_order:
---

挂代理，然后在wsl里面使用codex/cc cli是很常见的操作。这个过程中代理的一些问题时有发生。特意记录在此学习总结。

发现问题：
codex报错
Falling back from WebSockets to HTTPS transport.
stream disconnected before completion: Host is unreachable (os error 113)

error sending request for url:
https://chatgpt.com/backend-api/codex/responses

开发环境的网络有很多层，agent的网络访问是在sandbox里面还是外面，agent要用wsl的网络配置，wsl又是通过虚拟网卡经过windows主机中转，windows那边还配置了代理。所以只能一层层去查。既可以从外到内，也可以从内到外。但是外部问题比较好定位，只需要在windows打开chrome看能不能连上，就能排除代理问题。所以笔者通常从agent这边开始查，因为外部一般是好的。

先讲一下sandbox的问题。cli内部一般存在两个进程。一个是主进程，另一个是在需要执行测试等命令时创建的子进程。而为了保证测试的安全性，子进程会被放在一个隔离环境sandbox里面。抛开抽象的“容器”概念，sandbox从技术上可以理解为是在执行任意操作之前都要加上一系列限制，显然也包括网络。sandbox和主进程的网络是两个独立的上下文。所以在sandbox里面有可能访问不到远程的api，是很正常的。

接下来谈谈cli agent和wsl之间是如何进行网络交互的，或者说cli是如何使用wsl的网络。从最基本的os知识出发，cli是一个普通的wsl操作系统进程，所以使用的是wsl的网络协议栈。

wsl和windows之间的交互则是通过虚拟化实现的。正常来说，系统访问外部网络是通过网卡，网卡把数据转发到网关，也就是出外网之前必经的下一跳设备。在wsl里面，网卡和网关都是虚拟的。虚拟网关也是一个ip，记作wsl-gateway。wsl这侧看到的wsl-gateway就是他认为真实的网关。而在windows侧wsl-gateway则是一个被特殊标记的ip。在默认设置下windows会根据NAT规则把wsl-gateway转发到windows的真实网卡。当然这里的转发规则不止NAT，大同小异。所以这一部分主要需要检查一个事情：wsl能否解析自己的wsl-gateway对应的MAC地址。

再讲讲windows的代理机制。这里一般用的是clash，clash会在本地按不同协议开几个端口，其他本地windows进程会把自己的消息先转发给这些端口，clash把这些请求转发出去。比如说clash在HTTPS上监听127.0.0.1:7890，那么windows系统代理会被设置成127.0.0.1:7890。需要注意的是wsl不能直接访问127.0.0.1:7890因为在wsl和windows的网络上下文不同，这个ip:port代表的不是同一个local host。
