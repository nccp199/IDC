# 算法框架搭建——第十四部分：HAPPO＋MLP 连续 5-update 稳定性 Gate

## A. 阶段目标与执行边界

本阶段只验证 `HAPPO_MLP` 在固定配置下连续 5 次 update 的数值稳定性、顺序更新语义、有效动作 Mask、checkpoint/resume 和固定评估可重复性。

- 未实现 HGTA。
- 未修改 HAPPO/MAPPO 数学公式、环境、Reward、状态/观测/动作维度、BESS 物理映射、有效动作 Mask、Agent 顺序或训练超参数。
- 未启用 ValueNorm、Reward normalization、Grid Reward 或 Safe RL。
- 未运行第二条独立随机初始化的有效 5-update run，未运行 40-update。
- 只增加不参与 loss 的审计观测、artifact Gate 测试，以及一个 Windows TensorBoard 日志兼容修复。

最终结论：**通过，可以进入 HGTA 图结构设计阶段。** 本结论不包含 MAPPO/HAPPO 性能比较。

## B. 实际命令与唯一正式 Run

唯一产生 update 的全新正式命令：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe `
  train/train_harl_short.py `
  --algorithm happo `
  --critic-type mlp `
  --seed 7110 `
  --updates 5 `
  --episode-length 24 `
  --rollout-threads 2 `
  --device cpu `
  --output-dir runs/part14_happo_mlp_gate `
  --checkpoint-interval 1 `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path C:\Users\bulio\Desktop\IDC\ultimate_simplify\.tmp_harl_runtime
```

- exit code：0
- wall time：158.2 s
- 有效正式 run 数：1
- 输出目录：`runs/part14_happo_mlp_gate/gym/idc_bess_padding/happo/idc_bess_happo_short_HAPPO_MLP/seed-07110-2026-07-29-13-32-47`
- 方法：`algorithm_name=happo`、`critic_type=mlp`、`method_id=HAPPO_MLP`
- 固定配置：seed 7110、worker seeds `[7110, 8110]`、2 workers、24 steps、5 updates、CPU、torch threads 1、`share_param=false`、`fixed_order=true`、每 update checkpoint。

在此之前发生过一次命令包装层错误：外层执行器被错误设置为 1 秒 timeout，1.4 s 后返回 exit 124。检查确认无 Python 进程、无输出目录、无 rollout、无 update，因此不构成一条“正式有效训练”。错误输出已保留在任务执行记录，本报告不将其隐去。

## C. Update 1 零回归

Part 14 正式 update 1 与 Part 13 HAPPO smoke：

| 比较对象 | 结果 |
|---|---:|
| 48 行 Step，132 个公共非 run-ID 字段 | 差异 0 |
| 2 行 Episode，87 个公共非 run-ID 字段 | 差异 0 |
| 1 行 Update，69 个公共非时间字段 | 差异 0 |
| Actor 0 / Actor 1 / Critic 参数 | 最大差异 0 |
| 三个 optimizer 状态 | 最大差异 0 |
| ValueNorm/normalizer 状态 | 差异 0 |
| Python / NumPy / PyTorch RNG | 差异 0 |
| rollout action / action log-prob / critic reward / buffer | 最大差异 0 |
| logger 累计状态与 global step / episode 计数 | 差异 0 |

环境 checkpoint 中双方相同位置均含缓存诊断用 `NaN`，按 `equal_nan=true` 比较后其余值完全一致。`runner_state.total_updates_target` 分别为 1 和 5，是预期的目标长度配置差异，不是轨迹或训练状态差异。

结论：新增审计没有改变 update 1 的训练语义。

## D. 5-update 数据完整性

| 项目 | 实际值 | Gate |
|---|---:|---:|
| Step rows | 240 | 240 |
| Episode rows | 10 | 10 |
| Update rows | 5 | 5 |
| global step | 唯一连续 1–240 | 240 |
| completed updates | 5 | 5 |
| completed episodes | 10 | 10 |
| 每 worker、每 update 步数 | 24 | 24 |
| 每 update transitions | 48 | 48 |
| status | completed | completed |

没有缺失/重复 global step、重复 episode 或错序 update。正式 run 结束后 worker 生命周期正常，最终进程检查匹配 Python/HAPPO/MAPPO/IDC-BESS 的进程数为 0。

## E. HAPPO 更新顺序

五个 update 均满足：

- `agent_update_order=idc,bess`
- `first_updated_agent=idc`
- `second_updated_agent=bess`
- `actor_update_count_idc=1`
- `actor_update_count_bess=1`
- `critic_update_count=1`
- 三个 optimizer 的单 update step delta 均为 1，累计 step count 均为 1、2、3、4、5。

实现路径为：读取 rollout old log-prob → IDC 更新 → 重新计算 IDC new log-prob/ratio 并更新 factor → BESS 使用更新后 factor 训练 → 用 BESS 物理维 ratio 更新最终 factor → 两个 Actor 完成后 Critic 更新一次。

重新批量计算的 old log-prob 与 rollout buffer 已存 old log-prob 最大差为 `9.5367431640625e-07`，属于逐步前向与批量前向的 float32 运算次序误差；buffer 本身没有被覆盖，Part 13/14 和 continuous/resume 的 checkpoint 状态均逐元素一致。

actions、old log-probs、returns 和 advantages 的 update 内 mutation max diff 全部为 0；factor 每 update 新建为 `[24, 2, 1]` 的全 1 数组，没有跨 update 继承。

## F. Factor 逐 Update 统计与重构

| Update | after-IDC min | after-IDC max | final min | final max | after-IDC 重构差 | final 重构差 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.724011 | 1.538155 | 0.675560 | 1.502483 | 0 | 0 |
| 2 | 0.690843 | 1.196282 | 0.640265 | 1.217792 | 0 | 0 |
| 3 | 0.763561 | 1.140049 | 0.771627 | 1.152099 | 0 | 0 |
| 4 | 0.832523 | 1.167360 | 0.847579 | 1.194966 | 0 | 0 |
| 5 | 0.854824 | 1.161498 | 0.863908 | 1.160819 | 0 | 0 |

每个 checkpoint 保存 `initial`、`idc_effective_ratio`、`after_idc`、`bess_physical_ratio` 和 `final` 数组。离线以训练相同逐元素顺序重算：

```text
after_idc = initial * idc_effective_ratio
final = after_idc * bess_physical_ratio
```

五次重构最大差均严格为 0。每次 initial mean/min/max/p01/p50/p99 均为 1，initial exactly-one fraction 为 1；after-IDC exactly-one fraction 均为 0。所有 factor nonfinite count=0、nonpositive count=0，shape 始终 `[24, 2, 1]`，未发生错误广播或 44 维联合聚合。

## G. BESS 有效动作 Mask

五个 update 的以下指标均严格为 0：

- `bess_happo_factor_effective_physical_max_diff`
- `bess_happo_virtual_factor_contribution`
- `bess_virtual_mean_grad_norm`
- `bess_virtual_log_std_grad_norm`
- `bess_virtual_entropy_optimization_contribution`
- actor 内 `bess_effective_physical_ratio_max_diff`

BESS effective dim=1、padded dim=22、virtual dim=21。ratio、approximate KL、clip fraction、entropy 和 factor 只使用第 0 维；Bridge 继续只把第 0 维映射为物理充放电请求。虚拟输出可以随共享网络前向变化，但没有直接优化梯度或 factor/entropy 贡献。

## H. Actor 和 Critic 连续更新

| Update | IDC 参数 ΔL2 | BESS 参数 ΔL2 | Critic 参数 ΔL2 | IDC grad pre/post | BESS grad pre/post | Critic grad pre/post |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.045223 | 0.035339 | 0.048271 | 10.9304 / 10.0000 | 2.8303 / 2.8303 | 116.4409 / 10.0000 |
| 2 | 0.033812 | 0.031248 | 0.045293 | 10.6388 / 10.0000 | 2.3624 / 2.3624 | 126.8819 / 10.0000 |
| 3 | 0.025692 | 0.016867 | 0.043957 | 9.0963 / 9.0963 | 1.5153 / 1.5153 | 126.5911 / 10.0000 |
| 4 | 0.022722 | 0.013303 | 0.043184 | 9.1874 / 9.1874 | 0.6149 / 0.6149 | 133.5019 / 10.0000 |
| 5 | 0.020012 | 0.013242 | 0.042445 | 11.2073 / 10.0000 | 0.5392 / 0.5392 | 118.7503 / 10.0000 |

两个 Actor 与 Critic 每次参数 delta 均非零，loss、ratio、entropy、KL、梯度、value prediction、return、advantage 和 explained variance 全部 finite。Critic 原始梯度较大但按既有阈值裁剪；未启用 ValueNorm，未调整梯度阈值或学习率。

BESS rollout 物理动作范围为 `[-0.746196, 0.731311]`；五次 update 均有真实 BESS 参数变化。

## I. 环境、Reward、OPF 与 MEF

硬 Gate：

| 指标 | 结果 |
|---|---:|
| Reward reconstruction max error | 0 |
| OPF success rate | 1.0 |
| MEF success rate | 1.0 |
| action overflow count | 0 |
| nonfinite observation/state/action/reward count | 全 0 |
| voltage / line violation | 0 / 0 |
| safe violation sum / grid security penalty | 0 / 0 |
| Grid Reward / Safe RL | false / false |

240 step 总 reward 为 `-221.939039`。非零 Reward 分量总和包括：`r_done=30.278997`、`r_finished_tasks=9.0`、`r_priority=1.722691`、`r_cost=-8.672038`、`r_carbon=-29.359648`、`r_queue=-126.256703`、`r_overflow=-14.231948`、`r_urgent=-29.035669`、`r_waiting=-10.908858`、`r_deadline=-4.141935`、`r_peak_load=-5.048852`、`r_final_queue=-27.939668`、`r_final_soc=-3.296272`，其余详见 Step CSV。

工程指标：

- completed work `3,633,479.596`，finished tasks 186；episode final completion rate 均值 `0.407563`，final backlog 均值 `558,793.361`。
- cost sum `148,663.514`，carbon sum `146,798.241 kg`。
- BESS charge/discharge/throughput `67,701.527 / 53,650.420 / 121,351.947 kWh`。
- peak grid power `2,404.805 kW`，SOC 范围 `[0.1, 0.9]`，SOC boundary steps 5。
- invalid BESS request 19 steps，最大 infeasible power `653.446 kW`；这些请求由既有环境约束/惩罚处理，未导致越界、NaN、OPF/MEF 失败或 Reward 重构差异，列为后续监控而非本阶段算法阻塞。
- voltage 范围 `[1.014517, 1.089757] pu`，最大 line loading `1.253291%`。

以上只用于验证数据链稳定，不据 Reward 走势判断 HAPPO 相对性能。

## J. Checkpoint 完整性

存在并验证：

```text
update_000001.pt + manifest
update_000002.pt + manifest
update_000003.pt + manifest
update_000004.pt + manifest
update_000005.pt + manifest
final.pt + manifest
```

每个 manifest SHA-256 与实际文件一致，metadata 均为 `happo/mlp/HAPPO_MLP`；global step 依次 48/96/144/192/240，episode 依次 2/4/6/8/10。模型、三个 optimizer、normalizer、Python/NumPy/PyTorch RNG、两 worker 环境状态、rollout buffer、logger 累计状态、兼容性指纹和 factor audit 均存在。

`update_000005.pt` 与 `final.pt` 字节级相同，SHA-256 均为：

```text
1085f9d8ce205f92c28a03e83cca54af79b11ff5f5e47adb0cb49d0e8c7d375e
```

无 `.tmp` 残留，保存继续使用既有临时文件 + fsync + atomic replace/copy 语义。

## K. Update 3 Resume 精确一致性

恢复命令与正式配置相同，增加：

```powershell
--output-dir runs/part14_happo_mlp_resume_gate `
--resume-checkpoint runs/part14_happo_mlp_gate/.../checkpoints/update_000003.pt
```

首次恢复尝试在 update 4 的训练与结构化指标完成后、post-update checkpoint 前失败：

- exit code：1；wall time：46.1 s。
- 失败证据目录：`runs/part14_happo_mlp_resume_gate/.../seed-07110-2026-07-29-13-37-50`
- 原始问题：TensorBoardX 在 Windows 长路径下把 `critic/average_step_rewards` 重复作为子目录，抛出 `FileNotFoundError`。
- 根因：HARL upstream logger 使用 `add_scalars`，tensorboardX 为带 `/` tag 建嵌套 writer 目录；较长 resume 输出路径触发 Windows 路径问题。
- 修改文件：`marl/logging/training_metrics.py`。
- 修复：只在 upstream episode log 调用期间，把 `add_scalars` 映射为同值、同 global step 的 `add_scalar`；结构化 CSV/TensorBoard 主 logger 本来已经使用 `add_scalar`。
- 数学语义变化：否。模型、optimizer、RNG、环境、buffer、factor、loss 和超参数均未接触。
- 受影响测试：metrics composite logger、resume Gate、关键回归。
- 是否需重新执行完整 5-update Gate：否；正式 5-update 已成功且未受影响。只需从同一个正式 update 3 重新执行允许的恢复验证。

修复后恢复：exit code 0，wall time 142.2 s，输出 `seed-07110-2026-07-29-13-40-37`。

精确比较结果：

| 比较对象 | 结果 |
|---|---:|
| Step 145–240：96 行 × 132 非 run-ID 字段 | 差异 0 |
| Episode：4 行 × 87 非 run-ID 字段 | 差异 0 |
| Update 4–5：2 行 × 123 非时间字段 | 差异 0 |
| Actor 0/1、Critic、三个 optimizer | 差异 0 |
| normalizer、RNG、environment、buffer | 差异 0 |
| factor arrays/statistics/reconstruction | 差异 0 |
| logger 累计、global step、episode count | 差异 0 |

结论：update 3→5 精确恢复通过。

## L. 固定 Smoke 评估

使用 `smoke_v1 / scenario_000_seed_02026 / deterministic / 1 worker / 24 steps`，在一个评估命令中将 update 1、3、5 checkpoint 各加载两次：

- exit code：0
- wall time：170.1 s
- 输出：`evaluations/part14_happo_mlp_gate`
- Step rows：144；Episode rows：6；failure count：0
- actor load 成功，metadata 均为 `HAPPO_MLP/mlp`；actor 参数评估前后不变、gradient absent、eval mode=true。
- 所有动作 finite 且在 Box bounds 内；BESS Bridge 继续只消费第 0 维；OPF/MEF 全成功。

每个 checkpoint 的两次重复：24 个 Step 的全部 140 个非 model-ID 字段严格零差；Episode 除 `runtime_seconds` 外全部字段严格零差；aggregate 除 runtime metric 外严格零差。

单次工程 smoke 指标（仅记录，不做性能推断）：

| Update | Episode reward | Completion | Final backlog | Cost | Carbon kg |
|---:|---:|---:|---:|---:|---:|
| 1 | -16.518656 | 0.449585 | 433,763.299 | 14,013.547 | 13,665.195 |
| 3 | -16.833533 | 0.449099 | 434,146.519 | 14,611.987 | 14,200.888 |
| 5 | -16.987410 | 0.448535 | 434,591.042 | 14,868.240 | 14,429.192 |

## M. 测试结果

新增 `marl/tests/test_happo_mlp_five_update_gate.py`，覆盖 5-update 数据完整性、固定顺序/更新次数、factor reset/正值/重构、BESS 零贡献、真实参数/optimizer 更新、Part 13 update-1 零回归、update-3 resume、checkpoint/final、固定评估重复性和残留 child process。

| 命令/测试组 | exit | 结果 | wall time | warnings |
|---|---:|---|---:|---:|
| Part 14 artifact Gate | 0 | 9 passed | 12.3 s（pytest 6.34 s） | 0 |
| 指定六组关键回归 | 0 | 41 passed, 33 skipped | 183.7 s（pytest 177.70 s） | 832 |
| 前置 HAPPO 最小更新 | 0 | 3 passed | 7.9 s | 0 |
| 前置 Mask/metrics/checkpoint | 0 | 25 passed, 15 skipped | 123.6 s | 816 |

关键回归文件：

```text
test_unified_algorithm_interface.py
test_happo_mlp_update.py
test_bess_effective_action_mask.py
test_training_metrics_logger.py
test_training_checkpoint_resume.py
test_fixed_evaluation.py
```

832 条 warning 全部来自 pandapower `tap_dependency_table` 兼容数据的已知 DeprecationWarning；无测试失败。33 个 skip 是需要显式外部 artifact/可选运行条件的既有条件跳过，不是失败。未重复运行与本阶段改动无关且耗时很长的全历史物理测试；本阶段指定关键回归和专用 artifact Gate 已完整执行。

最终残留进程检查：`MATCHING_PROCESS_COUNT=0`。

## N. 问题分类

### 已修复、非算法阻塞

1. 正式命令第一次被执行包装器的 1 秒 timeout 提前终止；未创建 run、未产生 update。纠正执行时限后唯一有效正式 run 完成。
2. Resume 长路径触发 TensorBoardX `add_scalars` Windows 子目录错误；失败证据保留，改为日志等价的 `add_scalar` 后恢复精确零差。数学语义未变化。

### 后续继续监控

- Critic pre-clip gradient 约 116–134，既有 clip 后约 10；本阶段不调阈值、不启用 ValueNorm。
- update 1 factor 最宽约 `[0.6756, 1.5025]`，随后收窄；全程 finite/positive 且重构为 0，不触发调参。
- 19 个既有 BESS infeasible request、5 个 SOC boundary steps；环境正确处理且没有动作溢出或 OPF/MEF/Reward 链失败。
- old log-prob 批量复算与 rollout 存值最大 float32 差 `9.54e-07`；状态零回归和 resume 均严格为 0。

### 阻塞项

无。

## O. 第十四部分判断

| 判断项 | 结论 |
|---|---|
| HAPPO 连续更新 | 通过 |
| Agent 顺序和 factor | 通过，固定 IDC→BESS，重构差 0 |
| BESS 有效动作 Mask | 通过，相关零贡献全为 0 |
| Actor 更新 | 通过，两 Actor 每 update 真实更新一次 |
| Critic 更新 | 通过，每 update 更新一次 |
| 数值稳定性 | 通过，无 NaN/inf、factor 正且有限 |
| 环境与 Reward | 通过，重构差 0、OPF/MEF=1 |
| Checkpoint | 通过，1–5/final 完整且 hash 正确 |
| 精确 Resume | 通过，update 3→5 全训练状态差异 0 |
| 固定评估 | 通过，1/3/5 各两次精确复现 |
| MAPPO/统一接口兼容 | 通过，指定 registry/interface 回归通过 |

总体选择：**1. 通过，可以进入 HGTA 图结构设计阶段。**

## P. 下一阶段唯一建议

只进入 HGTA 图结构候选方案设计：整理候选节点类型、边/关系类型、节点特征、图批处理、HGTA 到 value head 的聚合方式，以及图状态 checkpoint/序列化接口；等待研究者确认方案后再实现。当前停止，不实现 HGTA，不运行 40-update，不做 MAPPO/HAPPO 性能比较。

