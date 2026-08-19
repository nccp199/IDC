# 算法框架搭建——第十七部分：HAPPO＋HGTA 连续 5 次更新稳定性验证

## A. 阶段目标和执行边界

本阶段只验证第 16 部分已接入的 `HAPPO_HGTA` 能否在冻结训练数学下连续完成 5 次更新，并验证逐次更新、checkpoint、update 3→5 精确恢复以及 actor-only 固定评估。

没有进行方法对比、性能判断、消融、调参或长训练。本文中的 reward/value loss 只作运行记录，不用于评价 HGTA 优劣。

## B. 代码冻结与运行配置

生产代码修改数量 = **0**。

本阶段只新增只读审计脚本 `marl/tests/audit_happo_hgta_part17.py`。该脚本只读取 checkpoint/CSV，并利用相邻 Adam 一阶矩反推出已进入 optimizer 的 post-clip gradient：

```text
g_t = (m_t - beta1 * m_(t-1)) / (1 - beta1)
```

它不构造训练环境、不调用 optimizer、不改变训练 artifact。脚本首次运行因 CSV 布尔值为 `True/False` 而非数字停止；只修正了该审计脚本的布尔解析，未写出失败审计结果，也未修改生产代码。

冻结正式配置：

```text
algorithm=happo
critic_type=hgta
method_id=HAPPO_HGTA
seed=7110
rollout_threads=2
episode_length=24
updates=5
device=cpu
checkpoint_interval=1
```

第 16 部分核心和 artifact 预检重新执行，`27 passed in 18.85s`；三个图 hash、1-update checkpoint 计数及统一接口均与第 16 部分一致。

由于生产代码修改数量为 0，本阶段未重复执行 `MAPPO_MLP` 或 `HAPPO_MLP` 零回归。

## C. 连续 5 次更新总体结果

唯一一次全新连续 5-update 正式训练成功：

```text
status                    = completed
termination_reason        = requested_updates_completed
total_updates_completed   = 5
total_environment_steps   = 240
episodes_completed        = 10
step rows                 = 240
episode rows              = 10
update rows               = 5
nan_or_inf_detected       = false
action_bounds_checks      = passed
```

五次更新的 global step/episode 为 `(48,2)、(96,4)、(144,6)、(192,8)、(240,10)`；两个 actor 和 critic 每次各更新一次，optimizer step 分别从 1 连续累积到 5。

正式 run：

```text
runs/part17_happo_hgta_5update_gate/gym/idc_bess_padding/happo/
idc_bess_happo_short_HAPPO_HGTA/seed-07110-2026-07-29-20-21-53
```

## D. 每次 HAPPO 更新顺序和修正因子

每次均记录 `first_updated_agent=idc`、`second_updated_agent=bess`，顺序始终为 IDC → BESS → critic。每次 factor 初始 min/mean/max 均精确为 `1/1/1`，不存在跨 update 继承。

| update | IDC 后 factor min / mean / max | 最终 factor min / mean / max | 非有限 / 非正 | IDC/最终重算误差 |
|---:|---|---|---:|---|
| 1 | 0.594702 / 0.998784 / 1.361538 | 0.589785 / 1.007043 / 1.479740 | 0 / 0 | 0 / 0 |
| 2 | 0.774970 / 0.988890 / 1.179592 | 0.750939 / 0.993874 / 1.246378 | 0 / 0 | 0 / 0 |
| 3 | 0.841609 / 1.021749 / 1.329495 | 0.834430 / 1.021553 / 1.381313 | 0 / 0 | 0 / 0 |
| 4 | 0.817943 / 0.985690 / 1.140003 | 0.781348 / 0.983570 / 1.227675 | 0 / 0 | 0 / 0 |
| 5 | 0.860695 / 1.013516 / 1.229072 | 0.856927 / 1.024759 / 1.248533 | 0 / 0 | 0 / 0 |

factor 的实际 buffer shape 仍由冻结 HAPPO 策略保持 `[24,2,1]`，对应单元测试和每次 48 样本统计一致；日志未发现遗漏、重复或换序更新。

## E. BESS 虚拟动作隔离

五次 update 的以下已有审计字段全部为 0：

```text
bess_virtual_mean_grad_norm
bess_virtual_log_std_grad_norm
bess_virtual_entropy_optimization_contribution
bess_effective_physical_ratio_max_diff
bess_happo_virtual_factor_contribution
```

checkpoint 级额外审计逐元素检查 `log_std[1:22]`、`fc_mean.weight[1:22]` 和 `fc_mean.bias[1:22]`：五次 update 的 `virtual_action_gradient_max_abs`、虚拟参数变化 max/norm 均精确为 0；有效第 0 维 gradient norm 依次为 `4.069811、4.236938、2.699848、3.865506、3.032944`，持续非零。

完整 240 step 中，执行映射 `physical = 2 × padded_dim0 - 1` 的最大误差为 0，证明动作执行只消费第 0 维。持久化日志没有名为 `virtual_action_ratio_contribution` 的独立列，但有效物理 ratio 重算误差和虚拟 factor 贡献均为 0，输出层虚拟行的梯度/参数变化亦为 0。

## F. HGTA 图结构恒定性

五个 update checkpoint 和 final 均满足：

```text
nodes=39, edges=90, node_types=7, relations=14
feature_schema_hash=1944df31af4e68360173cabea4f47d2d13ca8167e576fdb64ea2f76b1ed82b23
topology_hash=0a85394f385693afc351ec03831fd01bc1b75f31019e4f68bbd2c7bf5115c9f1
graph_schema_hash=32a642d211c7421634a7e4e65afb54529232ce4da19f47ddfd9fdf52e9ae4b26
```

节点/关系顺序、线路、变压器和归一化参考值逐 checkpoint 完全一致。冻结构造器无可变状态；B=1/2/48 的样本隔离与无串边在预检中再次通过。

## G. HGTA 逐模块梯度和参数变化

下表单元格为 `post-clip gradient norm / parameter delta norm`。update 1 的 delta 由 Adam step 1 精确公式重建；update 2～5 为相邻 checkpoint 参数直接相减。

| update | 类型投影 | 关系层 0 | 关系层 1 | forecast encoder | value head |
|---:|---|---|---|---|---|
| 1 | 1.145688 / 0.016800 | 1.091707 / 0.051218 | 0.623072 / 0.060940 | 0.202993 / 0.032608 | 9.852218 / 0.051766 |
| 2 | 1.174291 / 0.016165 | 1.105876 / 0.049438 | 0.633473 / 0.059089 | 0.212172 / 0.029972 | 9.846410 / 0.049877 |
| 3 | 1.153092 / 0.015697 | 1.081046 / 0.048595 | 0.626856 / 0.058351 | 0.242675 / 0.027988 | 9.851391 / 0.048480 |
| 4 | 1.134844 / 0.015181 | 1.050598 / 0.047154 | 0.608741 / 0.056812 | 0.304849 / 0.026525 | 9.856211 / 0.047670 |
| 5 | 1.108655 / 0.014759 | 0.994364 / 0.046080 | 0.586427 / 0.055872 | 0.284598 / 0.026017 | 9.866977 / 0.046976 |

固定模块规模和逐次非零 gradient tensor 数：

| 模块 | 参数 tensor | 参数数 | 每次有 gradient | 每次非零 gradient |
|---|---:|---:|---:|---:|
| 类型投影 | 14 | 1,664 | 14 | 14 |
| 关系层 0 | 91 | 36,568 | 91 | 76 |
| 关系层 1 | 91 | 36,568 | 91 | 76 |
| Forecast encoder | 4 | 11,360 | 4 | 4 |
| Value head | 4 | 16,513 | 4 | 4 |

所有梯度和 delta finite，五个模块每次变化均非零。两层中 14/14 relation-value 参数组每次均有非零梯度；9 个多入边 score 关系每次非零。五个单入边目标关系的 score 参数结构性为零，与第 16 部分解释一致，但其 relation-value 梯度非零，消息持续传递。

完整逐 tensor 数值见 `part17_readonly_audit.json`。

## H. Actor 逐次更新情况

IDC：

| update | 参数 delta | grad pre/post | policy loss | entropy | approx KL | clip fraction |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 0.046043 | 11.173674 / 10.000000 | -1.64e-7 | -8.185942 | 0.014515 | 0.229167 |
| 2 | 0.033089 | 9.243755 / 9.243755 | -1.74e-7 | -9.380794 | 0.005236 | 0.041667 |
| 3 | 0.025243 | 12.192199 / 10.000000 | 3.73e-7 | -8.860950 | 0.005587 | 0.062500 |
| 4 | 0.025610 | 11.267406 / 10.000000 | -3.68e-7 | -8.522936 | 0.002802 | 0 |
| 5 | 0.025349 | 10.248156 / 9.999999 | 4.66e-8 | -9.425923 | 0.004557 | 0.041667 |

BESS：

| update | 参数 delta | grad pre/post | policy loss | 有效熵 | approx KL | clip fraction |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 0.034246 | 4.071283 / 4.071283 | -0.095355 | -0.439435 | 0.000695 | 0 |
| 2 | 0.033335 | 4.239162 / 4.239162 | -0.049406 | -0.476830 | 0.000589 | 0 |
| 3 | 0.030610 | 2.701227 / 2.701227 | -0.056002 | -0.423887 | 0.000669 | 0 |
| 4 | 0.029937 | 3.868878 / 3.868878 | -0.031066 | -0.412591 | 0.000652 | 0 |
| 5 | 0.029730 | 3.039914 / 3.039914 | -0.040301 | -0.494338 | 0.000540 | 0 |

两 actor 每次参数变化均非零，全部 loss/gradient finite；runner 全程动作 Box 检查通过。表中 loss 正负不作性能解释。

## I. 价值网络数值稳定性

| update | value loss | value mean/std | return mean/std | advantage mean/std | explained variance | critic grad pre/post | critic delta |
|---:|---:|---|---|---|---:|---|---:|
| 1 | 55.082932 | -0.028743 / 0.071415 | -10.349080 / 3.343100 | -10.320338 / 3.355273 | -0.007296 | 53.461666 / 10.000003 | 0.101795 |
| 2 | 47.655376 | -0.241504 / 0.063928 | -9.673253 / 3.282296 | -9.431749 / 3.294777 | -0.007620 | 50.170914 / 10.000000 | 0.097893 |
| 3 | 33.640186 | -0.438089 / 0.054586 | -8.490258 / 1.555622 | -8.052168 / 1.562997 | -0.009504 | 49.936028 / 10.000001 | 0.095636 |
| 4 | 59.268131 | -0.635896 / 0.047215 | -11.448913 / 2.605956 | -10.813017 / 2.611717 | -0.004426 | 58.099388 / 10.000000 | 0.093046 |
| 5 | 51.198441 | -0.822205 / 0.038881 | -10.621339 / 3.437413 | -9.799134 / 3.438447 | -0.000602 | 52.776665 / 10.000000 | 0.091359 |

每次的 graph input、node embedding、attention、graph embedding、forecast embedding 和 value prediction 非有限计数均为 0。post-clip norm 仅有约 `2.8e-6` 的浮点容差，不超过现有阈值的合理误差；pre-clip norm 在 49.94～58.10 间有限且未无界增长。

冻结生产日志只持久化 value/return/advantage 的 mean/std，没有每次 min/max；`attention_nonfinite_count` 也没有把 score 与 weight 拆成两个列。网络内部对 score/value/softmax weight 均有 fail-fast finite 检查，5-update 无异常，但本阶段无法从既有 artifact 追溯三个 min/max或拆分后的两个计数。该可观测性缺口列为下一阶段监控项，不以伪造数值补齐。

## J. 环境与物理链结果

完整 240 step 审计：

```text
OPF success rate                 = 1.0
MEF success rate                 = 1.0
reward reconstruction max error = 0.0
shared reward max difference    = 0.0
numeric nonfinite count         = 0
action Box check failures       = 0
voltage violation count         = 0
line violation count            = 0
grid security penalty sum       = 0.0
BESS dim0 mapping max error      = 0.0
BESS padded dim0 range           = [0.116621, 0.873672]
BESS infeasible requests         = 31
max infeasible request power     = 1029.383421 kW
SOC boundary records             = 21
OPF cache hit rate               = 0.1125
MEF cache hit rate               = 0.0333333
logged overflow_work sum         = 7,097,380.872886
```

BESS 不可行请求均发生在合法动作范围内，由 SOC/能量/效率/功率约束正确投影，不等同动作越界。任务 overflow 是既有环境记录，本阶段不修改 reward 或环境，也不根据 5-update 内是否改善作判断。

## K. 检查点完整性

存在且可读：`update_000001.pt` 至 `update_000005.pt` 以及 `final.pt`。update checkpoint 大小依次为：

```text
2,007,429 / 2,095,621 / 2,182,981 / 2,258,053 / 2,336,069 bytes
```

每个 checkpoint 的 saved update、global step、episode 分别正确，包含两个 actor、HGTA critic、三个 optimizer、RNG、环境、runner、图元数据和兼容性信息。所有文件非空，manifest SHA-256 与文件对应。

`final.pt` 与 `update_000005.pt` 的 model、optimizer、normalizer、RNG、environment、runner、verification、graph metadata、compatibility 九个训练状态分区递归逐项完全相等。

## L. 第 3 次更新恢复到第 5 次更新的精确比较

恢复分支只产生 update 4、5；起点为 update 3、global step 144、episode 6，最终为 update 5、global step 240、episode 10。

连续 final 与恢复 final 的两个 actor、HGTA critic、三个 optimizer、normalizer、Torch/NumPy/Python RNG、environment、runner/counters、graph metadata/hash 和 compatibility 全部 exact。

恢复段 96 step、4 episode、update 4/5 的全部非计时训练字段 difference count=0、numeric max abs=0。verification 只差 run ID 与 rollout/update/wall/FPS 五类允许排除字段。因此结论是**精确恢复**，不是“仅能继续训练”。

审计文件：

```text
runs/part17_happo_hgta_resume_from_update3/.../part17_resume_exactness_audit.json
```

## M. update 1、3、5 固定 Actor 评估

一个正式 actor-only 命令加载 update 1、3、5 和 update 5 重复实例，在 `smoke_v1 / scenario_000_seed_02026 / 24 steps / CPU` 上得到：

```text
models=4, step rows=96, episode rows=4
status=completed, failure_count=0
critic_metrics_available=false
actor_parameters_unchanged=true
actor_gradients_absent=true
actors_in_eval_mode=true
```

四个 model manifest 均记录 `HAPPO_HGTA`、`critic_type=hgta` 及三个图 hash。actor-only loader 没有构造 HGTA，也没有加载 critic optimizer。

update 5 的两个持久化实例在排除 model ID/runtime 后，24 个 step 与 episode 全字段差异为 0。为直接检查完整动作向量，又用两个新环境进行两次内存级 actor-only 回放：`IDC action shape=[24,22]`，完整动作最大差异 0，24 个 BESS 物理动作最大差异 0，所有物理字段和 episode 汇总完全一致；参数未变、无梯度、eval mode 正确。

输出：`evaluations/part17_happo_hgta_actor_only`。

## N. 效率与资源记录

```text
完整外部进程墙钟                  300.958 s
run metadata 内部时间              268.765 s
pre-run 启动+导入+环境构造阶段       30.664 s
post-run 保存后退出                   1.529 s
五次 rollout 合计                  258.328 s
平均 rollout                       51.666 s/update
五次算法 update 合计                 1.095 s
平均算法 update                      0.219 s/update
恢复分支外部墙钟                    139.522 s
恢复 checkpoint load                 6.013 s
正式 actor-only 四实例外部墙钟      229.083 s
正式 evaluation 内部时间            219.465 s
```

现有日志只有整体 `update_time_seconds`，没有分别计时 IDC/BESS/critic；checkpoint manager 只在 metadata 留存最后一次 save `0.190879s`（恢复 final 为 `0.181986s`），没有逐 checkpoint save duration。pre-run 阶段也没有把 Python import 与环境初始化进一步拆分，因此这里只报告真实可得边界，不把它们伪拆成独立数值。

正式训练未预装进程峰值采样器。可得的只读资源指标为同时加载五个 update checkpoint 和 final 的进程 `PeakWorkingSetSize=218.520 MiB`；这不是正式训练 peak，不用于调参。

所有阶段结束后 `Get-Process python` 返回空，无残留训练、恢复、评估或测试进程。

## O. 实际执行命令和输出路径

冻结预检命令使用配置好的 Part 16 artifact 与 HARL `PYTHONPATH` 执行五个核心 pytest 文件；exit 0，27 passed，18.85s，0 training update。该辅助命令未用外层 wrapper 记录绝对开始/结束时刻，只保留 pytest/tool 墙钟。

唯一全新连续训练：

```powershell
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe train\train_harl_short.py `
  --algorithm happo --critic-type hgta --seed 7110 --updates 5 `
  --episode-length 24 --rollout-threads 2 --device cpu `
  --output-dir runs\part17_happo_hgta_5update_gate `
  --checkpoint-interval 1 `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path C:\Users\bulio\Desktop\IDC\ultimate_simplify\.tmp_harl_runtime
```

开始 `2026-07-29T20:21:28.9719956+08:00`，结束 `20:26:29.9296361+08:00`，300.958s，exit 0；产生唯一一组全新 update 1～5，输出路径见 C，无残留进程。

update 3→5 恢复：

```powershell
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe train\train_harl_short.py `
  --algorithm happo --critic-type hgta --seed 7110 --updates 5 `
  --episode-length 24 --rollout-threads 2 --device cpu `
  --output-dir runs\part17_happo_hgta_resume_from_update3 `
  --checkpoint-interval 1 --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path C:\Users\bulio\Desktop\IDC\ultimate_simplify\.tmp_harl_runtime `
  --resume-checkpoint <PART17_UPDATE_000003_PT>
```

开始 `2026-07-29T20:31:53.3409325+08:00`，结束 `20:34:12.8627032+08:00`，139.522s，exit 0；只产生 update 4、5，输出 `runs/part17_happo_hgta_resume_from_update3/.../seed-07110-2026-07-29-20-32-16`，无残留进程。

update 1/3/5/5 actor-only 固定评估：

```powershell
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe eval\eval_harl_fixed.py `
  --suite-id smoke_v1 --scenario-id scenario_000_seed_02026 `
  --checkpoint <UPDATE_1> --checkpoint <UPDATE_3> `
  --checkpoint <UPDATE_5> --checkpoint <UPDATE_5> `
  --output-dir evaluations\part17_happo_hgta_actor_only `
  --evaluation-id part17-happo-hgta-actor-only --device cpu `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path C:\Users\bulio\Desktop\IDC\ultimate_simplify\.tmp_harl_runtime
```

开始 `2026-07-29T20:35:45.3511553+08:00`，结束 `20:39:34.4346430+08:00`，229.083s，exit 0；0 training update，输出见 M，无残留进程。

只读审计命令：

```powershell
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe `
  marl\tests\audit_happo_hgta_part17.py <PART17_RUN> `
  --output <PART17_RUN>\part17_readonly_audit.json
```

有效重跑 exit 0，约 6.5s；0 training update。完整 update 5 动作重复审计 exit 0，约 125.4s；0 training update。两个辅助命令未用外层 wrapper 记录绝对开始/结束时刻，只保留 tool 墙钟，均无残留进程。

## P. 问题分类

### 阻塞问题

无训练、恢复、图结构、HAPPO、BESS、物理链、checkpoint 或 actor-only 评估阻塞。

### 下一阶段监控项

- 五次 critic pre-clip gradient 均触发/接近触发既有 norm=10 裁剪，但数值 finite 且无无界增长。
- 五个单入边目标关系的 score 参数结构性零梯度；其 relation-value 梯度持续非零。
- 31 次合法范围内 BESS 不可行请求、最大 1029.38 kW，以及 21 条 SOC 边界记录。
- 既有 logger 缺少 value/return/advantage min/max、score/weight 分列 nonfinite、actor/critic 分项时间和每次 checkpoint save 时间。
- HGTA/真实电网采样较慢；只记录，不在本阶段调参。

### 后续正式实验问题

HGTA 是否优于 MLP、reward 趋势、value loss 大小、任务 overflow、关系解释性与资源性价比，均需后续正式实验回答。本阶段不作结论。

## Q. 第十七部分最终判断

| 判断项 | 结论 |
|---|---|
| 连续 5 次更新完整性 | 通过 |
| HAPPO 更新顺序 | 通过，始终 IDC→BESS |
| 修正因子正确性 | 通过，每次从全 1 开始、finite/positive、重算误差 0 |
| IDC Actor 持续更新 | 通过，5/5 非零变化 |
| BESS Actor 持续更新 | 通过，5/5 非零变化 |
| BESS 虚拟动作隔离 | 通过，梯度/熵/ratio/factor/执行贡献证据均为 0 |
| HGTA 节点输入投影 | 通过，5/5 非零梯度和变化 |
| 第一层关系注意力 | 通过，5/5 持续更新 |
| 第二层关系注意力 | 通过，5/5 持续更新 |
| 未来预测编码器 | 通过，5/5 持续更新 |
| 类型级汇总和价值输出 | 通过，value head 5/5 持续更新 |
| 数值稳定性 | 通过，无 NaN/Inf；存在非阻塞可观测性缺口 |
| 图结构恒定性 | 通过，节点/边/类型/关系/hash 全程固定 |
| 环境物理链 | 通过，OPF/MEF=1、reward/shared diff=0、无越界 |
| 检查点完整性 | 通过，1～5+final 完整且 final=update5 |
| update 3 恢复精确性 | 通过，训练状态和非计时日志 bit-exact |
| 固定 Actor 评估 | 通过，1/3/5/5 全部成功且 update5 完整动作精确复现 |
| 代码冻结 | 通过，生产代码修改 0 |

总体选择：**2. 基本通过，可以进入第 18 部分，但有明确监控项。**

选择 2 而非 1 的唯一原因是冻结 logger 没有逐 update 的 value/return/advantage min/max、score/weight 分列计数及细分耗时；核心稳定性硬条件全部通过。该结论只代表连续训练框架可靠，不代表 HGTA 性能优于 MLP。

## R. 下一阶段唯一建议

仅进入第 18 部分三种方法统一验收，并在不改变算法数学、环境和超参数的前提下，把上述缺失的只读诊断字段纳入统一验收规范；不要同时启动长训练、性能结论、调参、消融、Safe RL 或新电网规模实验。
