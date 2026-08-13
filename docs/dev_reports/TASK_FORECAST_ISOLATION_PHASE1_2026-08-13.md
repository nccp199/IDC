# 第一阶段实现报告：未来任务真值隔离与可控 Forecast

日期：2026-08-13

修改前 HEAD：`e3e77c487a1e59ed6d891e5ae34fe4732c580754`

目标提交：`fix: isolate future task truth from MARL observations`

本阶段只修复未来任务信息泄漏并建立受控 task-arrival forecast。未修改 HGTA 节点、边、attention、relation、bus feature，未修改 MAPPO/HAPPO 更新、actor/critic 网络规模、reward/grid reward、Safe RL、BESS、AC OPF 或系统规模。

## 1. 修改前问题

### 1.1 泄漏数据流

修改前的数据流为：

```text
idc_model/task_model.py
IDCEnergyTaskModel.create_demo_tasks()
  -> reset 时预生成整个 episode 的真实 Task

IDCEnergyTaskModel.build_task_arrival_curve()
  -> 按真实 Task.arrival_time 和 Task.workload 汇总完整 24h lambda_t

envs/idc_price_env.py
IDCPriceEnv20D.reset()
  -> self.lambda_t = 完整 episode 真实到达曲线

IDCPriceEnv20D._get_forecast_features()
  -> 把 self.lambda_t 直接放入 144 维 forecast 的第 3 组

GridCoupledEnv / IDCGridMultiAgentEnv
  -> IDC local obs / BESS local obs / centralized state

marl/graphs/graph_builder.py
  -> state[:, 136:280] 进入 HGTA forecast branch
```

因此，IDC actor、BESS actor、MLP centralized critic 与 HGTA critic 都可以在 `t=0` 看到本 episode 后续真实随机任务到达 workload。CTDE 并不能使这种未来随机真值访问合法。

### 1.2 原 144 维 forecast 分类

144 维保持 `6 组 × 24 小时`：

| 组 | 内容 | 修改前语义 | 本阶段处理 |
|---|---|---|---|
| 1 | `price_t` | 当前场景按日内/日前已知价格使用 | 保持不变 |
| 2 | `T_amb` | 默认确定性合成曲线或外部整日序列；等价于 perfect temperature information | 保持不变，列为后续问题 |
| 3 | `lambda_t` | 从本 episode 真实未来 Task 汇总，属于 future truth leakage | 改为独立的 `task_arrival_forecast` |
| 4 | `pv_t` | 默认确定性/外部整日 PV 序列；当前仍相当于 perfect PV information | 保持不变，列为后续问题 |
| 5 | `time_sin` | 可合法提前知道的确定性时间编码 | 保持不变 |
| 6 | `time_cos` | 可合法提前知道的确定性时间编码 | 保持不变 |

逐项检查 Task 属性后确认：task-pool observation 只遍历已经到达/激活的任务；未到达 Task 的精确 deadline、priority、interruptible、parallelizable 没有单独进入 observation。修改前的直接泄漏集中在完整真实 `lambda_t`，它同时泄漏未来 arrival 与 aggregate workload。

## 2. 修改后的数据流

```text
真实 Task（reset 时预生成）
  -> true_task_arrival_profile
  -> 仅供环境按时激活任务、真实运行计算、离线误差评估
  -> 当前时刻已经发生的 arrival 可进入 current global feature

独立 forecast_rng
  + true_task_arrival_profile（仅用于受控仿真 forecast 合成）
  -> task_arrival_forecast（每个 episode reset 时生成一次并固定）
  -> 原 144 维 forecast 第 3 组
  -> IDC actor / BESS actor / centralized MLP critic
  -> HGTA 原 forecast branch
```

`lambda_t` 暂时保留为 ground-truth 兼容别名，以避免破坏既有环境内部/旧测试接口；未来观察构造器不再读取该别名。当前全局特征只读取 `true_task_arrival_profile[t]`，这是已经到达当前时刻的合法真值。

隔离测试会在 forecast 固定时修改所有尚未到达 Task 的 workload、deadline、priority、interruptible 与 parallelizable，并重建真实到达曲线；IDC actor、BESS actor 与 centralized state 均保持逐元素不变。

## 3. Forecast 定义

实现位置：`idc_model/task_forecast.py`。

支持模式：

- `noisy`：正式默认。一次性生成 `max(0, true[t] * (1 + error[t]))`，其中 `error[t] ~ Normal(0, forecast_error_level)`。
- `perfect`：`forecast = truth`，仅允许作为 oracle upper bound/debug，不是正式默认。
- `none`：全零 task-arrival forecast。

正式配置：

```text
task_forecast_mode = noisy
forecast_error_level = 0.20
task_forecast_seed_offset = 300000
```

RNG 规则：

```text
worker_seed = base_seed + worker_rank/stride 规则派生值
forecast_seed = worker_seed + task_forecast_seed_offset
```

Forecast 使用独立的 `numpy.random.Generator`，不消费 task RNG、server RNG、actor sampling RNG。`noisy` 在 metadata 中明确标为：

```text
synthetic forecast generated from ground truth with controlled error
```

它不是训练得到的真实 forecasting algorithm，也没有被包装成预测创新。

## 4. Observation 修改

维度与 HGTA 接口均未改变：raw base observation 仍为 280，Grid 后仍为 288；IDC obs 288，BESS true obs 164（再 pad 到 288），centralized state 294，HGTA forecast branch 144。

| 消费者 | 修改前 | 修改后 |
|---|---|---|
| IDC actor | 完整 144 维中包含 future task truth | 同一位置改为 24h `task_arrival_forecast` |
| BESS actor | 通过保留 forecast slice 看到 future task truth | 看到同一份合法 forecast |
| MLP centralized critic | centralized state 含 future task truth | 含同一份合法 forecast；没有额外 future truth |
| HGTA critic | `state[:, 136:280]` 的 forecast branch 含 future task truth | 切片、节点、边和 encoder 全不变，仅数据语义改为合法 forecast |

IDC/BESS 仍可看到当前状态、当前价格/温度/PV/arrival、已到达任务聚合、完整 24h 价格/温度/PV/时间假设以及 24h task forecast。两者都不能通过正式 observation 访问未到达 Task 的精确未来属性。

## 5. Checkpoint / exact resume

环境状态序列化新增：

- `true_task_arrival_profile`；
- `task_arrival_forecast`；
- forecast mode/error/seed；
- `forecast_rng.bit_generator.state`。

恢复时会核对 forecast 配置，恢复当前 episode 的 truth/forecast 和独立 RNG。测试同时比较了恢复当下数组与下一 episode 的 truth/forecast 序列。

环境 state 版本提升为 vec v3 / worker v2。缺少分离 forecast 的 legacy worker v1 被明确拒绝，避免把旧 `lambda_t` 静默恢复成 oracle forecast。固定评估场景 schema 提升为 v2，并在 artifact 中保存 mode/error/seed/source、MAE/RMSE/MAPE、truth 与 forecast。

## 6. Logging

日志 schema 提升到 metrics v2.1.0 / logger v3。新增：

- run metadata：mode、error level、forecast seed/rule/source、MAE/RMSE/non-zero MAPE、probe truth/forecast；
- step CSV：标量 forecast 配置和误差；truth/forecast 只在 terminal 行写入，避免每步重复大数组；
- episode CSV：完整 24 点 truth/forecast 与误差；
- fixed evaluation artifact：完整 forecast 定义、曲线和误差。

实际 HAPPO+HGTA smoke artifact 检查结果：mode=`noisy`、level=`0.2`、forecast seed=`307110`、episode truth/forecast 均为 24 点、OPF success rate=`1.0`。

## 7. 测试结果

| 验收项 | 结果 | 证据 |
|---|---|---|
| Test 1 future task isolation | PASS | 固定 forecast 后修改未到达任务真实属性，raw/current actor obs 不变 |
| Test 2 truth/forecast separation | PASS | noisy 下不相等；perfect 下相等；none 下为零 |
| Test 3 causality | PASS | 正式 IDC、BESS、centralized state 均逐元素不变 |
| Test 4 seed reproducibility | PASS | 同 seed truth/forecast 完全一致；不同 forecast seed 只改变 forecast；不同 task seed 改变 truth |
| Test 5 cross-method fairness | PASS | MAPPO+MLP、HAPPO+MLP、HAPPO+HGTA 的 probe truth/forecast 完全相同，seed 均为 307110 |
| Test 6 checkpoint resume | PASS | 当前 forecast/truth 和下一 episode forecast/truth 均一致 |
| Test 7 environment smoke | PASS | 正式双 Agent 完整 24 步；obs `(2,288)`、state `(2,294)`；reward/action finite；每步 OPF success |
| Test 8 MAPPO+MLP | PASS | 1 update / 24 steps；无 NaN/Inf；动作边界检查通过；checkpoint 生成 |
| Test 8 HAPPO+MLP | PASS | 1 update / 24 steps；无 NaN/Inf；动作边界检查通过；checkpoint 生成 |
| Test 8 HAPPO+HGTA | PASS | 1 update / 24 steps；无 NaN/Inf；动作边界检查通过；checkpoint 生成 |

最终测试命令与汇总：

```text
pytest marl/tests/test_task_forecast_isolation.py
  9 passed

pytest marl/tests/test_training_metrics_logger.py
  14 passed

pytest marl/tests/test_harl_mappo_formal_entry.py
  9 passed（包含正式 MAPPO 1-update）

pytest selected checkpoint + fixed evaluation generation
  2 passed

三种方法显式 CLI 1-update
  MAPPO_MLP   completed, 24/24 steps
  HAPPO_MLP   completed, 24/24 steps
  HAPPO_HGTA  completed, 24/24 steps
```

测试中只有 pandapower 的既有 spline/tap deprecation warnings；没有新增数值或 OPF 失败。

## 8. 未解决问题与 Git

未解决、且按范围要求没有顺手修改：

- PV 整日序列当前仍等价于 perfect/synthetic PV information，尚未拆分 `pv_true`/`pv_forecast`；
- temperature 整日序列仍为 perfect/synthetic temperature information；
- 当前 noisy task forecast 是 truth 加受控噪声的仿真产品，不是真实预测模型；
- price 继续按日前已知假设处理；若论文场景并非日前市场，需要后续单独澄清；
- `lambda_t` ground-truth 兼容别名仍保留，但已不再用于未来观察。

Git 处理：

- 修改前 HEAD：`e3e77c487a1e59ed6d891e5ae34fe4732c580754`；
- 修改前已有大量 modified/untracked 用户文件；全部保留；
- 本阶段只暂存 forecast 隔离、checkpoint/log/evaluation/test、正式入口 metadata 与本报告；
- `configs/config_ultimate.py` 中修改前已存在的 `cache_mef_load_bin_mw` 不纳入本阶段提交；
- 提交信息：`fix: isolate future task truth from MARL observations`；
- 最终 commit hash 与提交后 `git status` 见交付消息（提交 hash 不能自引用写入其自身内容）。
