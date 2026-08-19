# MAPPO短训练前检查——第八部分：训练日志与物理指标持久化

## 结论摘要

已在项目侧完成step、episode、update、TensorBoard四层日志，并完成正式24步、1-update验收。环境、Reward、MAPPO loss、GAE、有效动作Mask、Grid/Safe目标和训练超参数均未修改。

正式验收结果：

- `step_metrics.csv`：24行；
- `episode_metrics.csv`：1行；
- `update_metrics.csv`：1行；
- `metric_schema.json`、`run_summary.json`均存在且可解析；
- TensorBoard event存在，项目命名空间tag已写入；
- step Reward求和、episode Reward、update rollout Reward三者完全相等；
- 22个非alias Reward分量聚合最大误差`1.77635683940025e-15`；
- BESS ratio差、虚拟mean梯度、虚拟`log_std`梯度及虚拟entropy优化贡献均为0；
- OPF/MEF成功率均为1.0，Grid penalty与Safe violation均为0；
- 无NaN/inf。

第八部分判断：**2. 基本通过，可以训练但仍缺少少量诊断。** 缺口仅为物理环境当前未提供`ambient_temperature_C`和`server_load_max`；日志保留稳定列、显式availability标记和空值语义，没有估算或静默填0。该缺口不阻塞MAPPO短训练。

## A. 修改文件

### 新增

- `marl/logging/__init__.py`：导出结构化日志API和schema常量。
- `marl/logging/training_metrics.py`：实现`TrainingMetricsLogger`和`CompositeTrainingLogger`。
- `marl/tests/test_training_metrics_logger.py`：13项logger、terminal、Mask、多worker和CSV测试。
- `docs/dev_reports/MAPPO_SHORT_TRAINING_PRECHECK_PART8_LOGGING_PERSISTENCE.md`：本报告。

### 修改

- `marl/runners/idc_mappo_runner.py`：组合HARL原logger与项目logger，并在真实`train()`返回后写update指标。
- `train/train_harl_mappo_short.py`：补齐版本、Git dirty、数据hash、cache bin、日志metadata，并在正常/异常结束时生成`run_summary.json`。
- `configs/harl_mappo_short.yaml`：增加step、episode、TensorBoard日志开关；未改训练超参数。
- `marl/tests/test_harl_mappo_formal_entry.py`：增加24/1/1行数、Reward一致性、terminal、Mask、schema、summary和TensorBoard断言。

### 保持不动

- `envs/idc_price_env.py`、`env_wrappers/grid_coupled_env.py`：环境仍只返回`info`，不写文件。
- `marl/envs/idc_grid_multi_agent_env.py`、`marl/bridges/harl_bridge.py`、`marl/bridges/harl_padded_bridge.py`：环境和Bridge语义未改。
- `marl/algorithms/effective_action_mask.py`：MAPPO/HAPPO Mask语义未改。
- Reward、GAE、centralized critic、OPF、MEF、cache、Grid Reward、Safe RL及HARL仓库均未修改。

## B. 日志架构与调用链

```text
IDCPriceEnv20D.step()
  -> 原始Reward分量、任务、能源、成本、碳、BESS信息
GridCoupledEnv.step()
  -> OPF/LMP/MEF/电压/线路/Safe/cache字段；当前不改变Reward
IDCGridMultiAgentEnv.step()
  -> 同一个team reward复制给IDC/BESS；完整info继续传递
HarlIDCGridBridge.step()
  -> 为两个Agent复制info并增加agent_id/terminated/truncated
HarlPaddedBridge.step()
  -> 保持info；BESS padded 22维动作仍只切第0维进入物理层
ShareDummyVecEnv.step_wait()
  -> terminal时把terminal obs存入infos[worker][0].original_obs
  -> 自动reset只替换返回的下一观测，不覆盖terminal info
OnPolicyBaseRunner.run()
  -> logger.per_step(data)在buffer insert之前收到完整infos/rewards/actions
CompositeTrainingLogger.per_step()
  -> TrainingMetricsLogger.record_step()
  -> HARL原logger.per_step()
IDCOnPolicyMARunner.train()
  -> HARL真实actor/critic update
  -> TrainingMetricsLogger.record_update()
train_harl_mappo_short.run_training()
  -> 保存模型、metadata并finalize run_summary
```

| 层级 | 文件/函数 | 可获得的数据 | 是否适合记录 |
| --- | --- | --- | --- |
| 物理环境 | `envs/idc_price_env.py::IDCPriceEnv20D.step` | Reward、任务、功率、BESS、成本、碳 | 仅提供数据；不写文件 |
| Grid wrapper | `env_wrappers/grid_coupled_env.py::GridCoupledEnv.step` | OPF/LMP/MEF/Safe/cache | 仅提供数据；不写文件 |
| 双Agent环境 | `marl/envs/idc_grid_multi_agent_env.py::IDCGridMultiAgentEnv.step` | shared reward、完整info | 传递层，不写文件 |
| HARL Bridge | `marl/bridges/harl_bridge.py::HarlIDCGridBridge.step` | 两Agent info副本、terminal标志 | 传递层 |
| Padding | `marl/bridges/harl_padded_bridge.py::HarlPaddedBridge.step` | padded动作、完整info | 既有BESS虚拟动作监控保留 |
| VecEnv | `HARL/harl/envs/env_wrappers.py::ShareDummyVecEnv.step_wait` | terminal info及`original_obs` | 不持久化；保留自动reset边界 |
| HARL runner | `HARL/harl/runners/on_policy_base_runner.py::run` | `data=(obs,state,reward,done,infos,...,actions,log_prob,...)` | `per_step`是完整数据接入点 |
| 项目组合logger | `marl/logging/training_metrics.py::CompositeTrainingLogger` | 上述完整data及HARL原logger | 适合；不污染HARL |
| 结构化logger | `TrainingMetricsLogger` | step、episode accumulator、buffer和update诊断 | 实际CSV/TensorBoard写入点 |

当前正式入口仍限制`n_rollout_threads=1`，但logger按`worker_id`维护独立的`episode_id`和accumulator。两worker交错模拟测试已通过，没有共享单一episode状态。

## C. Step字段与单位

`step_metrics.csv`有133个稳定列。每一行对应一个worker的一次environment transition。

| 类别 | 主要字段 | 单位/语义 |
| --- | --- | --- |
| 索引 | `run_id, seed, update, global_step, worker_id, episode_id, episode_step, hour, terminated, truncated, is_terminal` | ID、计数、bool；`episode_step/hour=0..23` |
| Reward | `reward_returned, reward_total, base_reward, grid_adjusted_reward, grid_penalty` | Reward标量 |
| Reward分量 | `r_done, r_finished_tasks, r_priority, r_cost, r_carbon, r_queue, r_overflow, r_urgent, r_waiting, r_deadline, r_sla, r_unused, r_peak_load, r_pause, r_resume, r_noninterruptible, r_load_smooth, r_action_smooth, r_degradation, r_invalid_action, r_final_queue, r_final_soc` | 22个非alias分量 |
| Reward alias | `r_grid_peak` | `r_peak_load`别名；schema明确标注，不参与重构 |
| Reward校验 | `reward_reconstruction_error, idc_reward, bess_reward, shared_reward_abs_diff` | 误差/Reward |
| 任务 | `completed_work_step/total, finished_tasks_step/total, completion_rate, task_completion_rate, backlog_work, queue_work, overflow_work, urgent_backlog, active_task_count, waiting_time_mean, deadline_miss_step/total, sla_violation_count, sla_value, pause_count, resume_count, noninterruptible_violation_count` | work/count/hour/ratio |
| IDC | `idc_power_kW, it_power_kW, cooling_power_kW, pue, server_load_mean, grid_energy_kWh` | kW、ratio、kWh |
| 当前缺失 | `ambient_temperature_C, server_load_max`及对应`*_available` | 环境info不提供；CSV为空，不估算，不填0 |
| 成本/碳/峰值 | `price, carbon_factor, cost_step/total, carbon_step_kg/total, grid_power_kW, p_bus_net_kW, peak_power_kW, peak_excess_kW, pv_available/used_kW/kWh` | SGD相关量、kgCO2e、kW/kWh |
| Grid参考价格 | `grid_reference_usep_sgd_per_mwh` | NEMS USEP参考，不参与当前成本或Reward |
| BESS动作 | `bess_action_physical, bess_action_padded_dim0, bess_virtual_action_mean/std/min/max` | physical为Action Adapter后的有符号请求；padded dim0为HARL `[0,1]`输出 |
| BESS物理 | `bess_charge/discharge_power_kW, bess_soc, bess_energy_kWh, bess_charge/discharge_kWh, bess_throughput_kWh, bess_degradation_cost` | kW、ratio、kWh、成本 |
| BESS不可执行请求 | `bess_invalid_request, bess_infeasible_request_power_kW` | 物理请求与可执行功率差；不是Box越界 |
| Grid/Safe | `grid_load_scale, lmp, mef_plus/minus, voltage_min/max_pu, line_loading_max_pct, grid_loss_mw, opf/mef_success, voltage/line/opf_violation, safe_violation_total, safe_cost, grid_security_penalty` | MW、p.u.、%、bool/count；仅监控 |
| Cache | `opf/mef_cache_hit, opf/mef_cache_size, opf/mef_cache_hits/misses_total, cache_opf/mef_load_bin_mw` | bool/count/MW |
| 开关 | `grid_reward_enabled, safe_rl_enabled` | 本次均为false |

必须字段缺失会`KeyError`；关键数值NaN/inf会`FloatingPointError`；Reward重构或returned/adjusted Reward不一致会fail-fast。solver失败时可能不存在的LMP/MEF/电压/线路数值写空值，其success flag仍为权威状态，schema明确空值语义。

当前价格语义已区分：`price`是进入IDC成本和Reward的人工价格；`grid_reference_usep_sgd_per_mwh`只是NEMS USEP参考。

## D. Episode聚合规则

`episode_metrics.csv`有88个稳定列。只写完整24步episode；异常退出时step CSV保留，不完整episode不写入正式episode CSV，并在`run_summary.json.incomplete_episode_steps_by_worker`记录。

| Episode字段 | Step来源 | 聚合 |
| --- | --- | --- |
| `episode_reward` | `reward_returned` | sum；只计一次team reward |
| `average_step_reward` | `reward_returned` | mean |
| `terminal_reward` | `reward_returned` | final |
| `r_*_sum` | 对应22个非alias `r_*` | sum |
| `completed_work_sum, finished_tasks_sum` | step增量 | sum |
| `final_completion_rate, final_task_completion_rate` | completion字段 | final |
| `mean_queue, max_queue, final_backlog` | queue/backlog | mean/max/final |
| `waiting_time_mean` | `waiting_time_mean` | mean |
| deadline/SLA/pause/resume | step增量或累计字段 | sum或final，按语义区分 |
| `grid_energy_kWh_sum, cost_sum, carbon_kg_sum, pv_used_kWh_sum` | step物理量 | sum |
| `mean_pue` | `pue` | mean |
| `mean/max_grid_power_kW, max_peak_excess_kW` | 功率/峰值 | mean/max |
| `mean_bess_soc, final_bess_soc` | `bess_soc` | mean/final |
| BESS charge/discharge/throughput/degradation | 对应step kWh/cost | sum |
| BESS action mean/std | `bess_action_physical` | mean/std |
| charge/discharge/idle fraction | charge/discharge power | rate |
| OPF/MEF/cache hit | bool | rate |
| 电压/线路 | voltage/line字段 | min/max/mean/rate |
| `safe_violation_sum/max` | `safe_violation_total` | sum/max |
| final totals/cache sizes | 对应累计字段 | final |
| `shared_reward_max_diff` | 两Agent Reward差 | max |

没有对所有数值统一求平均。

## E. Update训练字段

`update_metrics.csv`有54个稳定列，每次真实policy update写一行。

| 类别 | 字段 |
| --- | --- |
| 状态/性能 | `update, global_step, episodes_completed, wall_time_seconds, rollout_time_seconds, update_time_seconds, steps_per_second` |
| rollout | `rollout_episode_reward_mean, rollout_step_reward_mean` |
| Critic | `value_loss, critic_grad_norm, value_prediction_mean, return_mean, advantage_mean, advantage_std, explained_variance` |
| IDC Actor | `idc_policy_loss, idc_entropy, idc_actor_grad_norm, idc_ratio_mean/std/min/max, idc_clip_fraction, idc_approx_kl, idc_action_mean/std` |
| BESS Actor | `bess_policy_loss, bess_effective_entropy, bess_actor_grad_norm, bess_effective_ratio_mean/std/min/max, bess_effective_clip_fraction, bess_effective_approx_kl, bess_physical_action_mean/std` |
| Mask | `bess_effective/padded/virtual_action_dim, bess_effective_physical_ratio_max_diff, bess_virtual_mean_grad_norm, bess_virtual_log_std_grad_norm, bess_virtual_entropy_optimization_contribution, bess_mask_enabled` |
| Shared Reward | `idc_reward_mean, bess_reward_mean, shared_reward_max_diff` |
| Grid rollout | `opf_success_rate, mef_success_rate, safe_violation_sum` |

Critic扩展指标直接由当前buffer的`value_preds`和`returns`计算，不改变return、GAE或critic更新。

正式1-update样例：

| 指标 | 值 |
| --- | ---: |
| value loss | 82.437255859375 |
| critic grad norm | 139.9957275390625 |
| IDC policy loss | -4.967053879312289e-09 |
| BESS policy loss | -3.476937493473997e-08 |
| IDC entropy | -8.970846176147461 |
| BESS effective entropy | -0.5529008507728577 |
| IDC ratio mean | 0.9368818402290344 |
| BESS effective ratio mean | 0.9984898567199707 |
| BESS physical action mean | 0.5063628281156222 |
| rollout time | 43.61887140000181s |
| update time | 0.024096400000416907s |
| throughput | 0.5499167726168883 steps/s |

## F. TensorBoard tags

项目tag写入run根`logs/events.out.tfevents.*`；二进制event实际检出以下命名空间：

- `reward/episode_total`, `reward/step_mean`及全部非alias `reward/r_*`；
- `task/completion_rate`, `task/completed_work`, `task/final_backlog`, `task/deadline_miss`, `task/sla_violation`；
- `energy/cost`, `energy/carbon_kg`, `energy/grid_kWh`, `energy/peak_power_kW`, `energy/pv_used_kWh`；
- `bess/soc_final`, `bess/soc_mean`, `bess/charge_kWh`, `bess/discharge_kWh`, `bess/throughput_kWh`, `bess/invalid_request`, `bess/physical_action_mean`；
- `grid/opf_success_rate`, `grid/mef_success_rate`, `grid/voltage_min_pu`, `grid/line_loading_max_pct`, `grid/lmp_mean`, `grid/mef_mean`, `grid/safe_violation_total`；
- `train/value_loss`, `train/critic_grad_norm`, `train/idc_policy_loss`, `train/bess_policy_loss`, `train/idc_entropy`, `train/bess_effective_entropy`, `train/idc_clip_fraction`, `train/bess_clip_fraction`；
- `mask/bess_effective_ratio_diff`, `mask/bess_virtual_mean_grad`, `mask/bess_virtual_log_std_grad`, `mask/bess_virtual_entropy_contribution`。

HARL原有agent/critic TensorBoard tag继续保留。CSV与项目TensorBoard聚合使用相同字段语义。

## G. Terminal/自动reset测试

- 第23小时terminal info在`logger.per_step(data)`中完整可用；
- 正式CSV最后一行为`hour=23, episode_step=23, is_terminal=True`；
- `r_final_queue=-4.18775379835698`只进入当前episode一次；
- `r_final_soc=-0.0935093275825871`只进入当前episode一次；
- VecEnv把terminal observation保存到`original_obs`后自动reset，但logger不读取它作为普通next observation；
- reset后hour 0由下一次transition以新的`episode_id`记录；
- 单元测试确认episode CSV每24步恰好一行；
- incomplete episode不写episode CSV，summary显式记录残留step数；
- 两worker交错模拟得到两个独立episode，不混合。

## H. Mask诊断日志

正式update长期写入：

```text
bess_effective_action_dim = 1
bess_padded_action_dim = 22
bess_virtual_action_dim = 21
bess_effective_physical_ratio_max_diff = 0
bess_virtual_mean_grad_norm = 0
bess_virtual_log_std_grad_norm = 0
bess_virtual_entropy_optimization_contribution = 0
bess_mask_enabled = true
```

首个update执行完整梯度诊断。当前实现仍每update记录完整诊断；40-update规模下开销相对于OPF很小，后续如扩展大规模并行可再配置诊断周期，本部分未改变Mask算法。

## I. Grid/Safe与cache日志

正式episode：

| 指标 | 值 |
| --- | ---: |
| OPF成功率 | 1.0 |
| MEF成功率 | 1.0 |
| Grid reward penalty累计 | 0.0 |
| Safe violation累计 | 0.0 |
| min/max voltage | 1.0144508162 / 1.0895998655 p.u. |
| max line loading | 1.2531355734% |
| OPF cache hit rate | 0.0 |
| MEF cache hit rate | 0.0 |
| OPF load bin | 0.1 MW |
| MEF load bin | 0.01 MW |

cache hit率为0是本次首次24小时访问的实际结果，不是缺失值。Grid/Safe字段完整持久化，但`grid_reward_enabled=false`、`safe_rl_enabled=false`，没有进入PPO目标。

## J. 输出目录示例

```text
runs/part8_logging_acceptance/gym/idc_bess_padding/mappo/
└── idc_bess_mappo_short/seed-07110-2026-07-28-19-09-09/
    ├── config.json
    ├── resolved_config.json
    ├── run_metadata.json
    ├── models/
    │   ├── actor_agent0.pt
    │   ├── actor_agent1.pt
    │   └── critic_agent.pt
    ├── logs/
    │   ├── events.out.tfevents.*
    │   ├── summary.json
    │   └── HARL原有agent/critic event目录
    └── metrics/
        ├── step_metrics.csv
        ├── episode_metrics.csv
        ├── update_metrics.csv
        ├── metric_schema.json
        └── run_summary.json
```

主要文件大小：step CSV 36,920 bytes；episode CSV 2,763 bytes；update CSV 1,868 bytes；schema 80,918 bytes；summary 6,056 bytes。训练结束后仅依赖run目录即可重建Reward、任务、能源、BESS、Grid/Safe、cache和训练曲线。

## K. 性能开销

相同seed 7110、相同固定0.5动作、两个独立24步环境episode的墙钟对比：

| 模式 | 24步耗时 |
| --- | ---: |
| 完全不创建logger | 35.0641997s |
| step+episode CSV启用、TensorBoard关闭 | 39.1783674s |
| 单次观测差异 | +4.1141677s / +11.7332% |

由于每次AC OPF/MEF求解存在数秒级墙钟波动，上述单次对照超过10%。为确认差异来源，又隔离测量了纯logger Python/CSV路径：处理2,400个transition耗时0.5894446s，折合每个24步episode为`0.00589445s`，仅占正式rollout 43.6189s的`0.01351%`。因此结构化日志不是11.73%差异的来源，也不是OPF训练瓶颈。

实现保持文件持续打开；每个episode结束时flush；不使用DataFrame；不读取旧CSV；不每step计算hash；不每stepfsync。Grid CSV hash只在启动metadata阶段计算一次。

## L. 测试结果

统一使用：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:PYTHONPATH=((Get-Location).Path + ';' + $env:HARL_SOURCE_PATH)
```

| 测试 | 真实pytest参数 | 结果 | warnings | 耗时 |
| --- | --- | ---: | ---: | ---: |
| 新logger测试 | `-m pytest marl/tests/test_training_metrics_logger.py -q` | 13 passed, 0 failed | 16 | 5.61s |
| Mask快速回归 | `-m pytest marl/tests/test_bess_effective_action_mask.py -q -k 'not four_virtual_action_patterns'` | 11 passed, 0 failed, 1 deselected | 0 | 4.02s |
| Mask完整24步物理回归 | `-m pytest marl/tests/test_bess_effective_action_mask.py -q -k 'four_virtual_action_patterns'` | 1 passed, 0 failed, 11 deselected | 800 | 159.86s |
| 正式入口及日志一致性 | `-m pytest marl/tests/test_harl_mappo_formal_entry.py -q` | 9 passed, 0 failed | 257 | 75.05s |
| 既有MAPPO smoke | `-m pytest marl/tests/test_harl_mappo_smoke.py -q -s` | 1 passed, 0 failed | 201 | 60.44s |
| Grid/cache正式回归 | `-m pytest marl/tests/test_grid_cache_formal_entry.py -q` | 11 passed, 0 failed | 56 | 113.23s |

warning均为pandapower既有弃用警告、smoke中的既有tensor转换提示，以及沙箱无法写`.pytest_cache`提示。

中间失败记录：

- 新logger测试首次为`10 passed, 3 failed`：据真实环境修正了`bess_raw_action`与padded dim0的不同语义、双方同为NaN的共享字段比较，以及测试中的上一update收尾；没有修改环境或动作映射。最终13项全部通过。
- 正式入口首次在受限沙箱setup阶段出现9个dependency API error，因为`.tmp_harl_runtime`目录ACL不可读；未执行测试逻辑。使用同一既有运行时的获批只读访问后通过，未联网或安装依赖。

本部分没有重新运行Bridge/padding组合，因为要求列出的五组回归已全部覆盖，且第七部分完整物理不变性回归已再次通过。

## M. 1-update验收

正式命令：

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  train/train_harl_mappo_short.py `
  --config configs/harl_mappo_short.yaml `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path .tmp_harl_runtime `
  --seed 7110 `
  --updates 1 `
  --episode-length 24 `
  --rollout-threads 1 `
  --output-dir runs/part8_logging_acceptance `
  --scenario idc_bess_padding `
  --device cpu
```

退出码0，总墙钟69.2s，完成1次update、24环境步、1个完整episode。

### 文件与行数

| 文件 | 数据行数/状态 |
| --- | ---: |
| `step_metrics.csv` | 24 |
| `episode_metrics.csv` | 1 |
| `update_metrics.csv` | 1 |
| `metric_schema.json` | 可解析，schema `1.0.0` |
| `run_summary.json` | 可解析，status `completed` |
| TensorBoard events | 13个event文件，含1个项目根event |

### 一致性

```text
sum(step.reward_returned) = -27.0993857467547
episode.episode_reward    = -27.0993857467547
update.rollout_reward     = -27.0993857467547
三者最大误差             = 0

Reward分量episode/step聚合最大误差 = 1.77635683940025e-15
每步Reward重构最大误差             = 0
shared_reward_max_diff              = 0
```

### 关键物理样例

```text
final_completion_rate      = 0.3072096948
final_task_completion_rate = 0.4838709677
final_backlog              = 837550.7596713955
final_bess_soc             = 0.4032453362
cost_sum                   = 14345.212494826848
carbon_kg_sum              = 13836.264994062563
grid_energy_kWh_sum        = 20983.750977516655
max_grid_power_kW          = 2064.7257917300444
bess_charge_kWh_sum        = 5081.448316574097
bess_discharge_kWh_sum     = 5505.17641172541
bess_throughput_kWh_sum    = 10586.624728299506
```

`nan_or_inf_detected=false`；关键训练、Reward和物理字段均finite。可选缺失字段使用空CSV值，不伪装为0。

## N. 问题分类

### 阻塞MAPPO短训练

- 无。

Reward没有双倍累计，terminal没有错位，Mask诊断持续有效，CSV三层一致，日志没有改变训练语义。

### 必须在40-update前修正

- 无当前阻塞项。

现有日志足以分析3 seeds × 40 updates的Reward、任务、BESS、成本、碳、Grid/Safe、cache和actor/critic曲线。

### 建议改进

- 如后续确需温度/服务器极值分析，由环境在独立任务中正式暴露`ambient_temperature_C`和`server_load_max`；当前不得估算。
- 大规模并行后可增加Parquet或异步写线程；当前24步短训练没有必要。
- 大规模训练后可把完整Mask梯度诊断改为首个update加固定周期；当前每update记录仍很轻量。
- 后续checkpoint/复现部分再处理源码快照、Git diff归档和resume语义。

### 可以保持现状

- 环境只提供info、logger在项目runner侧持久化；
- step CSV持续打开，episode/update边界flush；
- per-worker accumulator；
- terminal `original_obs`处理；
- Reward alias与重构校验；
- Grid/Safe只监控；
- BESS有效动作Mask诊断；
- HARL原TensorBoard日志与项目tag并存。

## O. 第八部分判断

**选择2：基本通过，可以训练但仍缺少少量诊断。**

理由：训练、Reward、BESS、Grid/Safe、cache及actor/critic数据已经结构化、持久化并完成正式1-update一致性验收；仅`ambient_temperature_C`和`server_load_max`因现有环境info不提供而显式留空。这两个字段不影响训练语义、Reward重构、episode边界或短训练分析主链。

## P. 最小后续建议

1. 可按既定计划进入下一部分检查；本次不要为两个可选物理字段改环境。
2. 首次5-update/40-update前保留当前`logger.step_logging_enabled=true`以便诊断；确认稳定后可按磁盘需求关闭step CSV，episode/update日志必须保持开启。
3. 训练后以`run_summary.json`定位run状态和日志路径，以`metric_schema.json`解释字段，不直接依赖代码猜测。
4. 后续实现checkpoint/resume时，明确新run目录或严格schema/run_id校验，继续禁止无提示append旧CSV。

本报告到此停止，不进入第九部分，也未运行5-update或40-update训练。
