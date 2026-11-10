# RTX5090 四组实验

协议：`rtx5090-flux-taco-arrival-four-arm-v1`；完成范围：`train`。

使用Flux GemmV2 DP模板主循环，修改CTA排程与完成发布；单次完整GEMM，并通过NCCL传输就绪块。不是原Flux ReduceScatter实现，也不宣称原kernel逐字未变。

完整训练窗口：32；累计正式 optimizer step：1600（各窗口独立初始化）。

独立到达校准计划：`a9536769d7a571a0138b2064b1ed26d0195e381d853bc48340084535c702405b`。

| 角色 | 到达生产顺序identity | 改变槽位 |
| --- | --- | ---: |
| attn | False | 16 |
| mlp | True | 0 |

identity只描述生产排程；D仍采用每lane就绪消费，identity回退如实保留。校准不计入正式operator/训练窗口；每个trial共用本批冻结计划。

## operator / trial1 / attn

单位：us/call。每窗口取所有 rank 最大墙钟耗时。

| 组 | 均值 | 中位数 | 最慢窗口 |
| --- | ---: | ---: | ---: |
| flux | 2338.725381 | 2262.733680 | 2793.807709 |
| flux_taco | 2948.867971 | 2927.673635 | 3549.978635 |
| flux_taco_interleaved | 2881.749817 | 2893.246216 | 2985.771394 |
| flux_taco_interleaved_arrival | 2708.718269 | 2701.586762 | 2964.920935 |

| 组 | 独立完整GEMM均值 μs | 最大BF16相对L2 | 最大codec参考相对L2 |
| --- | ---: | ---: | ---: |
| flux | 92.823784 | 0 | 0 |
| flux_taco | 89.138864 | 0.024688427 | 0.0001728754 |
| flux_taco_interleaved | 82.794045 | 0.024688427 | 0.0001728754 |
| flux_taco_interleaved_arrival | 82.545156 | 0.024688427 | 0.0001728754 |

| 人工错峰诊断 | 观测到绕过的rank窗口 / 诊断rank窗口 |
| --- | ---: |
| flux_taco_interleaved_arrival | 8/64 |

该诊断使用独立实例与人工lane延迟，不进入正式计时。未观测到绕过也如实报告；观测到绕过不代表普通负载获得加速。

| 前者 / 后者 | 配对下降 % | 95% CI | 更快块数 |
| --- | ---: | --- | ---: |
| flux_taco/flux | -26.0946 | [-32.1347, -20.5008] | 0/8 |
| flux_taco_interleaved/flux_taco | +1.9431 | [-3.5455, +7.2084] | 6/8 |
| flux_taco_interleaved_arrival/flux_taco_interleaved | +6.1088 | [+2.0678, +9.7412] | 6/8 |
| flux_taco_interleaved/flux | -23.6445 | [-30.8679, -16.3354] | 0/8 |
| flux_taco_interleaved_arrival/flux | -16.0913 | [-25.4521, -6.5874] | 2/8 |

## operator / trial1 / mlp

单位：us/call。每窗口取所有 rank 最大墙钟耗时。

| 组 | 均值 | 中位数 | 最慢窗口 |
| --- | ---: | ---: | ---: |
| flux | 2413.839282 | 2313.428753 | 2906.613851 |
| flux_taco | 2985.563569 | 2975.810334 | 3386.095096 |
| flux_taco_interleaved | 2905.571371 | 2819.011613 | 3261.892556 |
| flux_taco_interleaved_arrival | 2653.354392 | 2643.533230 | 2793.444244 |

| 组 | 独立完整GEMM均值 μs | 最大BF16相对L2 | 最大codec参考相对L2 |
| --- | ---: | ---: | ---: |
| flux | 194.206766 | 0 | 0 |
| flux_taco | 194.543334 | 0.024672596 | 0.00017910955 |
| flux_taco_interleaved | 194.234773 | 0.024672596 | 0.00017910955 |
| flux_taco_interleaved_arrival | 194.248001 | 0.024672596 | 0.00017910955 |

| 人工错峰诊断 | 观测到绕过的rank窗口 / 诊断rank窗口 |
| --- | ---: |
| flux_taco_interleaved_arrival | 8/64 |

该诊断使用独立实例与人工lane延迟，不进入正式计时。未观测到绕过也如实报告；观测到绕过不代表普通负载获得加速。

| 前者 / 后者 | 配对下降 % | 95% CI | 更快块数 |
| --- | ---: | --- | ---: |
| flux_taco/flux | -23.7884 | [-35.0536, -13.0790] | 1/8 |
| flux_taco_interleaved/flux_taco | +2.5594 | [-1.4016, +5.0742] | 7/8 |
| flux_taco_interleaved_arrival/flux_taco_interleaved | +8.5767 | [+4.2587, +12.7666] | 7/8 |
| flux_taco_interleaved/flux | -20.6202 | [-30.8080, -10.5617] | 1/8 |
| flux_taco_interleaved_arrival/flux | -10.2750 | [-15.1855, -4.9758] | 1/8 |

## train / trial1 / full_step

单位：ms/step。每窗口取所有 rank 最大墙钟耗时。

| 组 | 均值 | 中位数 | 最慢窗口 |
| --- | ---: | ---: | ---: |
| flux | 1470.338779 | 1465.823582 | 1492.996653 |
| flux_taco | 1606.202685 | 1607.987172 | 1618.975498 |
| flux_taco_interleaved | 1595.270132 | 1596.608458 | 1613.794172 |
| flux_taco_interleaved_arrival | 1535.577973 | 1526.861942 | 1572.401860 |

| 组 | tokens/s | 平均loss | 末步loss均值 | 平均grad norm | 末步grad norm均值 | peak allocated GiB | peak reserved GiB | allocated＋native GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flux | 5571.505 | 3.2622016 | 7.3457602e-05 | 5.6021828 | 0.0065914303 | 17.9727 | 18.6152 | 17.9805 |
| flux_taco | 5100.228 | 2.9549329 | 0.00011870055 | 5.1695127 | 0.0028737335 | 18.0274 | 18.6406 | 18.0352 |
| flux_taco_interleaved | 5135.180 | 2.9549329 | 0.00011870055 | 5.1695127 | 0.0028737335 | 18.0274 | 18.6406 | 18.0352 |
| flux_taco_interleaved_arrival | 5334.799 | 2.9549329 | 0.00011870055 | 5.1695127 | 0.0028737335 | 18.0274 | 18.6367 | 18.0352 |

| 前者 / 后者 | 配对下降 % | 95% CI | 更快块数 |
| --- | ---: | --- | ---: |
| flux_taco/flux | -9.2428 | [-10.1803, -8.2859] | 0/8 |
| flux_taco_interleaved/flux_taco | +0.6800 | [+0.1603, +1.1220] | 6/8 |
| flux_taco_interleaved_arrival/flux_taco_interleaved | +3.7476 | [+2.9457, +4.4914] | 8/8 |
| flux_taco_interleaved/flux | -8.4999 | [-9.6027, -7.4345] | 0/8 |
| flux_taco_interleaved_arrival/flux | -4.4338 | [-5.6174, -3.6519] | 0/8 |

正下降率表示前者更快。CI按完整块bootstrap，未做多重比较校正；rank与窗口内step不作为独立样本。不同trial分别报告。

loss和grad norm仅取rank0的正式步，不重复计TP rank；末步值先取每窗口最后一步，再跨窗口平均。operator张量scratch包含于Torch峰值，不重复相加；原生GEMM cudaMalloc工作区另按rank加至allocated后取最大。allocated与reserved不相加，CUDA context和NCCL内部占用未包含。

完整GEMM独立计时不含通信/归约，不代替四组完整投影或训练比较。可选profile已核对文件哈希；通信event区间包含等待，报告不据此声称已验证kernel重叠。

配置、源码快照、命令、逐rank结果、失败信息和可选profile保存在批次目录。
