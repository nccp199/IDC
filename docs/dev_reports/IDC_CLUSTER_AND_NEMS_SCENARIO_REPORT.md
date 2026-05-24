# IDC Cluster And NEMS Scenario Report

## 1. 修改文件列表

- `configs/config_ultimate.py`
- `idc_model/power_model.py`
- `idc_model/task_model.py`
- `envs/idc_price_env.py`
- `env_wrappers/grid_coupled_env.py`
- `eval/eval_base.py`
- `train/train_ppo_ultimate.py`
- `scripts/smoke_test_grid_coupled_env.py`
- `scripts/build_grid_load_scale_from_nems.py`
- `data/grid_scenarios/nems_singapore/raw/`
- `data/grid_scenarios/nems_singapore/processed/`

## 2. 服务器集群建模说明

本次将原来的 20 个可控服务器单元升级为 20 个 server group。每个 group 内部包含 100 台同构服务器，group 之间继续继承原有异构差异。

- `num_server_groups = 20`
- `server_group_size = 100`
- `effective_total_server_count = 2000`
- PPO 动作空间保持不变，仍为 20 个 server group 控制量加 3 个额外控制量，`action_dim = 23`
- 不展开为 2000 个单机对象，也不扩大动作空间

## 3. 处理能力和功耗同步缩放

`IDCPowerModel` 先生成单机级异构参数，再按 `server_group_size` 放大为 group 级参数：

- `group_capacity = single_server_capacity * server_group_size`
- `group_idle_power = single_server_idle_power * server_group_size`
- `group_max_power = single_server_max_power * server_group_size`

同时新增运行信息字段：

- `server_group_model_enabled`
- `server_group_size`
- `num_server_groups`
- `effective_total_server_count`
- `total_group_capacity`
- `total_group_idle_power_kW`
- `total_group_max_power_kW`

## 4. 任务压力同步缩放

新增 `IDC_SCALE_CONFIG["task_workload_scale"] = 100`。任务 workload 使用单机基准 IDC capacity 乘以 `task_workload_scale`，避免与 group capacity 放大重复相乘，同时让任务压力与 100 倍服务器规模匹配。

`Q0`、`lambda_ref`、`queue_ref`、`queue_capacity_ref`、`sla_ref` 在环境层随 `task_workload_scale` 缩放，保持状态归一化和任务 reward normalizer 的数量级稳定。

新增 info 字段：

- `task_workload_scale`
- `effective_total_workload`
- `average_task_workload`

## 5. BESS 同步缩放

新增 `IDC_SCALE_CONFIG["bess_scale_factor"] = 100`，且 `scale_bess_with_idc=True` 时：

- `bess_capacity_kWh *= 100`
- `bess_charge_power_max_kW *= 100`
- `bess_discharge_power_max_kW *= 100`

SOC 上下限比例、充放电效率和 no sell-back 逻辑保持不变，reward 权重未修改。

新增/确认每步 info 字段：

- `bess_mode`
- `bess_charge_power_kW`
- `bess_discharge_power_kW`
- `bess_soc`
- `bess_capacity_kWh`
- `bess_charge_power_max_kW`
- `bess_discharge_power_max_kW`
- `bess_available_charge_kWh`
- `bess_available_discharge_kWh`
- `bess_charge_efficiency`
- `bess_discharge_efficiency`
- `bess_cycle_throughput_kWh`
- `bess_degradation_cost`
- `bess_scale_factor`

## 6. NEMS CSV 处理脚本

新增脚本：

- `scripts/build_grid_load_scale_from_nems.py`

运行方式：

```bash
python -m scripts.build_grid_load_scale_from_nems
python -m scripts.build_grid_load_scale_from_nems --input data/grid_scenarios/nems_singapore/raw/RT72_EGO_23May2026.csv
```

脚本自动识别列名：

- `Date`
- `Period`
- `Demand (MW)`
- `USEP ($/MWh)`

支持列名空格、大小写和括号差异。不会联网下载，也不会写爬虫。

## 7. 原始与处理目录

原始 CSV 放置目录：

- `data/grid_scenarios/nems_singapore/raw/`

处理结果输出目录：

- `data/grid_scenarios/nems_singapore/processed/`

输出文件：

- `data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale.csv`
- `data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale_meta.json`

当前检查时，`data/grid_scenarios/nems_singapore/raw/RT72_EGO_23May2026.csv` 尚不存在，因此 processed CSV/metadata 未生成。脚本已给出清晰错误。

## 8. grid_load_scale 计算方式

脚本选择拥有完整 48 个半小时 period 的日期，将半小时数据聚合到 24 小时：

- period 1-2 -> hour 0
- period 3-4 -> hour 1
- ...
- period 47-48 -> hour 23

每小时：

- `demand_mw` 为两个半小时 Demand 的平均
- `usep_sgd_per_mwh` 为两个半小时 USEP 的平均
- 默认 `grid_load_scale = demand_mw / daily_mean_demand`
- `--scale-method max` 时为 `demand_mw / daily_max_demand`

## 9. GridCoupledEnv 动态电网负荷接入

新增 `GRID_SCENARIO_CONFIG`：

- `enable_dynamic_grid_load=True`
- `grid_load_scale_path=data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale.csv`
- `grid_load_scale_column=grid_load_scale`
- `grid_usep_column=usep_sgd_per_mwh`
- `fallback_load_scale=1.0`

`GridCoupledEnv` 初始化时尝试读取 24 小时 `grid_load_scale_t`。每一步按 `hour % 24` 取：

```python
load_scale_t = grid_load_scale_t[hour % 24]
```

并传入：

```python
solve_opf(..., load_scale=load_scale_t)
calculate_nodal_mef(..., load_scale=load_scale_t)
```

新增 info 字段：

- `grid_load_scale`
- `grid_scenario_enabled`
- `grid_scenario_source`
- `grid_scenario_message`
- `grid_reference_usep`

## 10. USEP 与 OPF LMP 的关系

`grid_reference_usep` 只作为真实市场价格参考，不替代 IEEE14 OPF 求解得到的 `grid_lmp`。后续可以用 USEP 做外部价格场景对照，但当前电网物理反馈仍来自 `grid_model.solve_opf()`。

## 11. Smoke Test 结果

运行命令：

```bash
python -m py_compile scripts/build_grid_load_scale_from_nems.py env_wrappers/grid_coupled_env.py scripts/smoke_test_grid_coupled_env.py configs/config_ultimate.py
python -m scripts.build_grid_load_scale_from_nems --input data/grid_scenarios/nems_singapore/raw/RT72_EGO_23May2026.csv
python -m scripts.smoke_test_grid_coupled_env
```

实际环境中 `python` 不在 PATH，使用 bundled Python 执行；同时已安装缺失依赖 `gymnasium` 和 `pandapower`。

结果摘要：

- `obs_dim = 264`
- `action_dim = 23`
- `server_group_size = 100`
- `effective_total_server_count = 2000`
- `task_workload_scale = 100`
- `bess_capacity_kWh = 10000`
- `bess_charge_power_max_kW = 2000`
- `bess_discharge_power_max_kW = 2000`
- `grid_load_scale = 1.0` fallback
- `grid_reference_usep = NaN`，因为 NEMS processed CSV 尚未生成
- `grid_opf_success_count = 24`
- `grid_mef_success_count = 24`
- `grid_opf_fail_count = 0`
- `grid_mef_fail_count = 0`
- `avg_grid_lmp = 40.166847`
- `avg_grid_mef_plus = 168.421475`
- `min_grid_voltage = 1.015573`
- `max_grid_line_loading = 1.235871`
- `reward_mismatch_count = 0`

## 12. 当前仍未改 reward 的原因

本次只做建模尺度升级和动态电网场景接口，不改 grid reward 权重。`enable_grid_reward=False` 仍为默认值，因此：

- `grid_adjusted_reward == base_reward`
- 电网反馈进入 info 和 grid_obs
- 不改变当前 PPO reward 语义

## 13. 后续建议

- 放入真实 NEMS CSV 后重新运行处理脚本，检查 `grid_load_scale` 是否有合理日内波动
- 检查不同 `server_group_size`
- 检查不同 `bess_scale_factor`
- 打开 grid reward ablation，但先保持权重实验可控
- 后续接入 `forecast_grid_obs`
- 后续 SAFE RL
- 后续 MAPPO+CTDE
