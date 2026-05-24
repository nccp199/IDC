# NEMS Multiday Scenario Update Report

## 1. 修改文件列表

- `scripts/build_grid_load_scale_from_nems.py`
- `scripts/smoke_test_grid_coupled_env.py`
- `docs/dev_reports/NEMS_MULTIDAY_SCENARIO_UPDATE_REPORT.md`

本次未修改 `algorithms/`、训练逻辑、reward 权重、服务器集群建模、BESS 缩放逻辑或 `grid_model` OPF/MEF 核心逻辑。

## 2. 新支持的数据文件

脚本现在同时支持：

- `RT72_EGO_23May2026.csv`
- `USEP_May-2026.csv`

`USEP_May-2026.csv` 是多日半小时数据文件，每天应包含 period 1 到 48，可作为后续动态电网负荷场景的主数据源。

## 3. 列名自动识别

脚本对列名做大小写不敏感和符号清洗：

- `DATE` / `Date`
- `PERIOD` / `Period`
- `DEMAND (MW)` / `Demand (MW)`
- `USEP ($/MWh)` / `USEP`

如果存在 Solar 字段，会在单日输出中附加 `solar_mw`；Solar 不是必需字段。

## 4. 完整日识别

脚本按日期分组，检查每个日期是否覆盖完整 period 集合：

```text
1, 2, ..., 48
```

只使用完整日。metadata 会记录：

- `all_dates`
- `complete_dates`
- `incomplete_dates`
- `complete_day_count`
- `selected_date`
- `selection_method`

如果没有完整日，会给出清晰错误。

## 5. selection 参数

新增 `--selection`，默认 `typical`：

- `first-complete`：选择第一个完整 48 period 日期
- `typical`：选择日均 demand 最接近所有完整日均值的日期
- `peak-demand`：选择日最大 demand 最高的日期
- `peak-price`：选择日最大 USEP 最高的日期
- `low-demand`：选择日均 demand 最低的日期

如果指定 `--target-date`，则优先使用目标日期，并要求该日期完整。

## 6. 半小时转小时

聚合规则：

- period 1-2 -> hour 0
- period 3-4 -> hour 1
- ...
- period 47-48 -> hour 23

每小时：

- `demand_mw` 为两个半小时 DEMAND 平均
- `usep_sgd_per_mwh` 为两个半小时 USEP 平均
- `solar_mw` 如果存在则为两个半小时 Solar 平均

## 7. grid_load_scale 计算

`--scale-method` 支持：

- `mean`：`grid_load_scale = demand_mw / selected_day_mean_demand`
- `max`：`grid_load_scale = demand_mw / selected_day_max_demand`
- `global-mean`：`grid_load_scale = demand_mw / all_complete_days_mean_demand`

默认使用 `mean`。

## 8. 输出文件

单日主场景：

- `data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale.csv`

metadata：

- `data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale_meta.json`

如果指定 `--export-all-days`：

- `data/grid_scenarios/nems_singapore/processed/nems_all_days_hourly_profiles.csv`

all-days profile 可用于多场景评估、代表日选择、未来 `forecast_grid_obs` 设计和不同电网日类型对照。

## 9. USEP 与 OPF LMP

USEP 只作为 `grid_reference_usep` 写入 `GridCoupledEnv.info`，用于真实市场价格参考。它不替代 IEEE14 OPF 求解得到的 `grid_lmp`。

GridCoupledEnv 仍按小时使用：

```python
load_scale_t = grid_load_scale_t[hour % 24]
solve_opf(..., load_scale=load_scale_t)
calculate_nodal_mef(..., load_scale=load_scale_t)
```

## 10. 检查结果

编译检查通过：

```bash
python -m py_compile scripts/build_grid_load_scale_from_nems.py scripts/smoke_test_grid_coupled_env.py env_wrappers/grid_coupled_env.py configs/config_ultimate.py
```

真实数据处理命令已运行：

```bash
python -m scripts.build_grid_load_scale_from_nems --input data/grid_scenarios/nems_singapore/raw/USEP_May-2026.csv --selection typical --export-all-days --verbose
```

当前工作区中该 raw 文件尚不存在，因此脚本给出清晰错误：

```text
NEMS CSV not found: .../data/grid_scenarios/nems_singapore/raw/USEP_May-2026.csv
```

为验证多日逻辑，使用临时三日样本做了离线自检：

- 识别 3 个完整日
- `typical` 选择中间日期
- 成功生成单日 `nems_24h_load_scale.csv`
- 成功生成 `nems_all_days_hourly_profiles.csv`
- `grid_load_scale` 出现日内波动
- 临时测试文件已清理

## 11. Smoke Test 结果

运行：

```bash
python -m scripts.smoke_test_grid_coupled_env
```

结果摘要：

- `obs_dim = 264`
- `action_dim = 23`
- `grid_scenario_enabled = True`
- 当前 processed CSV 不存在，因此使用 fallback `grid_load_scale = 1.0`
- `grid_reference_usep = NaN`
- OPF success `24/24`
- MEF success `24/24`
- `reward_mismatch_count = 0`

用户放入 `USEP_May-2026.csv` 并运行处理脚本后，smoke test 应读取 processed CSV，不再 fallback，且 `grid_reference_usep` 应不再全为 NaN。

## 12. 后续建议

- 使用不同 `--selection` 做多场景评估
- 用 `peak-demand` 和 `peak-price` 构造压力测试日
- 后续 `forecast_grid_obs` 可使用 USEP/Demand 的日前预测
- 后续做 grid reward ablation
- 后续 SAFE RL
- 后续 MAPPO+CTDE
