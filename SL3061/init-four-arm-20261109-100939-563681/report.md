# RTX5090 四组实验

协议：`rtx5090-flux-taco-arrival-four-arm-v1`；完成范围：`train`。

使用Flux GemmV2 DP模板主循环，修改CTA排程与完成发布；单次完整GEMM，并通过NCCL传输就绪块。不是原Flux ReduceScatter实现，也不宣称原kernel逐字未变。

完整训练窗口：32；累计正式 optimizer step：1600（各窗口独立初始化）。

独立到达校准计划：`cd6301fb2a7c6f61778c4004fd70c746400d32e4ba4b6b33a0b3cc63d3e370b0`。

| 角色 | 到达生产顺序identity | 改变槽位 |
| --- | --- | ---: |
| attn | True | 0 |
| mlp | True | 0 |

identity只描述生产排程；D仍采用每lane就绪消费，identity回退如实保留。校准不计入正式operator/训练窗口；每个trial共用本批冻结计划。

## operator / trial1 / attn

单位：us/call。每窗口取所有 rank 最大墙钟耗时。

| 组 | 均值 | 中位数 | 最慢窗口 |
| --- | ---: | ---: | ---: |
| flux | 2365.797020 | 2380.961243 | 2498.840806 |
| flux_taco | 2842.935234 | 2868.188316 | 3155.474684 |
| flux_taco_interleaved | 2813.774829 | 2804.652746 | 2951.584531 |
| flux_taco_interleaved_arrival | 2643.724161 | 2679.450102 | 2920.271936 |

| 组 | 独立完整GEMM均值 μs | 最大BF16相对L2 | 最大codec参考相对L2 |
| --- | ---: | ---: | ---: |
| flux | 90.029956 | 0 | 0 |
| flux_taco | 85.169579 | 0.024688427 | 0.0001728754 |
| flux_taco_interleaved | 99.341589 | 0.024688427 | 0.0001728754 |
| flux_taco_interleaved_arrival | 96.227662 | 0.024688427 | 0.0001728754 |

| 人工错峰诊断 | 观测到绕过的rank窗口 / 诊断rank窗口 |
| --- | ---: |
| flux_taco_interleaved_arrival | 8/64 |

该诊断使用独立实例与人工lane延迟，不进入正式计时。未观测到绕过也如实报告；观测到绕过不代表普通负载获得加速。

| 前者 / 后者 | 配对下降 % | 95% CI | 更快块数 |
| --- | ---: | --- | ---: |
| flux_taco/flux | -20.0375 | [-26.4946, -13.2575] | 0/8 |
| flux_taco_interleaved/flux_taco | +0.9074 | [-2.7089, +5.0112] | 5/8 |
| flux_taco_interleaved_arrival/flux_taco_interleaved | +6.2141 | [+0.0882, +11.1940] | 7/8 |
| flux_taco_interleaved/flux | -18.9483 | [-25.3397, -12.7414] | 0/8 |
| flux_taco_interleaved_arrival/flux | -11.5567 | [-18.2350, -5.0993] | 1/8 |

## operator / trial1 / mlp

单位：us/call。每窗口取所有 rank 最大墙钟耗时。

| 组 | 均值 | 中位数 | 最慢窗口 |
| --- | ---: | ---: | ---: |
| flux | 2355.987432 | 2361.317223 | 2481.884237 |
| flux_taco | 2920.773911 | 2969.311490 | 3139.485761 |
| flux_taco_interleaved | 2810.030240 | 2818.803085 | 2955.359307 |
| flux_taco_interleaved_arrival | 2598.828952 | 2525.362326 | 3068.080894 |

| 组 | 独立完整GEMM均值 μs | 最大BF16相对L2 | 最大codec参考相对L2 |
| --- | ---: | ---: | ---: |
| flux | 194.053607 | 0 | 0 |
| flux_taco | 195.253674 | 0.024672596 | 0.00017910955 |
| flux_taco_interleaved | 194.007081 | 0.024672596 | 0.00017910955 |
| flux_taco_interleaved_arrival | 194.074593 | 0.024672596 | 0.00017910955 |

| 人工错峰诊断 | 观测到绕过的rank窗口 / 诊断rank窗口 |
| --- | ---: |
| flux_taco_interleaved_arrival | 8/64 |

该诊断使用独立实例与人工lane延迟，不进入正式计时。未观测到绕过也如实报告；观测到绕过不代表普通负载获得加速。

| 前者 / 后者 | 配对下降 % | 95% CI | 更快块数 |
| --- | ---: | --- | ---: |
| flux_taco/flux | -23.8431 | [-29.9446, -17.5136] | 0/8 |
| flux_taco_interleaved/flux_taco | +3.7194 | [+1.0187, +6.2550] | 6/8 |
| flux_taco_interleaved_arrival/flux_taco_interleaved | +7.8271 | [+1.8338, +12.8834] | 5/8 |
| flux_taco_interleaved/flux | -19.2369 | [-22.9958, -15.0235] | 0/8 |
| flux_taco_interleaved_arrival/flux | -9.9040 | [-17.0744, -3.5610] | 1/8 |

## train / trial1 / full_step

单位：ms/step。每窗口取所有 rank 最大墙钟耗时。

| 组 | 均值 | 中位数 | 最慢窗口 |
| --- | ---: | ---: | ---: |
| flux | 1464.504915 | 1465.727696 | 1482.588006 |
| flux_taco | 1607.684668 | 1605.029011 | 1637.465299 |
| flux_taco_interleaved | 1593.048003 | 1596.432533 | 1615.247695 |
| flux_taco_interleaved_arrival | 1524.970043 | 1525.057439 | 1534.898991 |

| 组 | tokens/s | 平均loss | 末步loss均值 | 平均grad norm | 末步grad norm均值 | peak allocated GiB | peak reserved GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flux | 5593.699 | 3.2622016 | 7.3457602e-05 | 5.6021828 | 0.0065914303 | 17.9727 | 18.6152 | 17.9805 |
| flux_taco | 5095.527 | 2.9549329 | 0.00011870055 | 5.1695127 | 0.0028737335 | 18.0274 | 18.6367 | 18.0352 |
| flux_taco_interleaved | 5142.343 | 2.9549329 | 0.00011870055 | 5.1695127 | 0.0028737335 | 18.0274 | 18.6367 | 18.0352 |
| flux_taco_interleaved_arrival | 5371.909 | 2.9549329 | 0.00011870055 | 5.1695127 | 0.0028737335 | 18.0274 | 18.6367 | 18.0352 |

| 前者 / 后者 | 配对下降 % | 95% CI | 更快块数 |
| --- | ---: | --- | ---: |
| flux_taco/flux | -9.7746 | [-10.6738, -8.8817] | 0/8 |
| flux_taco_interleaved/flux_taco | +0.9118 | [+0.2830, +1.4767] | 7/8 |
| flux_taco_interleaved_arrival/flux_taco_interleaved | +4.2693 | [+3.6027, +4.7842] | 8/8 |
| flux_taco_interleaved/flux | -8.7737 | [-9.4699, -8.0186] | 0/8 |
| flux_taco_interleaved_arrival/flux | -4.1298 | [-4.5529, -3.6933] | 0/8 |

正下降率表示前者更快。CI按完整块bootstrap，未做多重比较校正；rank与窗口内step不作为独立样本。不同trial分别报告。

loss和grad norm仅取rank0的正式步，不重复计TP rank；末步值先取每窗口最后一步，再跨窗口平均。operator张量scratch包含于Torch峰值，不重复相加；原生GEMM cudaMalloc工作区另按rank加至allocated后取最大。allocated与reserved不相加，CUDA context和NCCL内部占用未包含。

完整GEMM独立计时不含通信/归约，不代替四组完整投影或训练比较。可选profile已核对文件哈希；通信event区间包含等待，报告不据此声称已验证kernel重叠。

配置、源码快照、命令、逐rank结果、失败信息和可选profile保存在批次目录。
