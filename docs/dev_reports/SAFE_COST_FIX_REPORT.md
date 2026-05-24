# SAFE_COST Fix Report

## 1. 修改文件列表

本次代码修复涉及：

- `grid_model/grid_case.py`
- `grid_model/opf_solver.py`
- `grid_model/grid_metrics.py`
- `env_wrappers/grid_coupled_env.py`
- `eval/eval_base.py`
- `scripts/smoke_test_grid_coupled_env.py`

验证时按用户要求运行了 `scripts.scan_ieee14_bus_sensitivity`，该脚本刷新了以下生成结果文件：

- `data/grid_node_sensitivity/ieee14_node_sensitivity_static.csv`
- `data/grid_node_sensitivity/ieee14_node_sensitivity_detail.json`

本次没有修改 `algorithms/`，没有修改 reward 权重，没有训练 PPO，也没有启用 grid reward。

## 2. 修复背景

上一轮审计发现，`scripts.smoke_test_grid_coupled_env` 中 `total_safe_cost = 194` 全部来自 `safe_cost_voltage`。当时：

- OPF success: `24/24`
- MEF success: `24/24`
- `safe_cost_line = 0`
- `safe_cost_opf = 0`
- 最大线路负载率约 `1.25%`
- smoke test 只打印 `minV`，没有打印 `maxV`

根因是 `grid_model/grid_metrics.py` 使用硬编码全局电压阈值 `0.95/1.05` 判断 voltage violation，而 pandapower IEEE14 网络实际 bus 电压约束不是统一 `0.95/1.05`。

当前 IEEE14 case 经 `harmonize_generator_voltage_limits()` 后，实际约束类似：

- `min_vm_pu = 0.94`
- `max_vm_pu in {1.06, 1.07, 1.09}`

因此 OPF 满足 pandapower bus-specific limit 时，旧逻辑仍可能把 `1.08-1.09` 的正常可行电压误判为 `> 1.05` 的安全违规。

## 3. OPFResult 新增真实约束字段

`grid_model/grid_case.py` 的 `OPFResult` 新增：

```python
bus_voltage_min_pu: dict[int, float]
bus_voltage_max_pu: dict[int, float]
line_loading_limit_percent: dict[int, float]
```

`GridMetricResult` 额外新增：

```python
voltage_violation_magnitude: float
line_overload_magnitude: float
```

## 4. OPF solver limit 提取

`grid_model/opf_solver.py` 现在从 pandapower net 中提取真实约束：

- `net.bus["min_vm_pu"] -> bus_voltage_min_pu`
- `net.bus["max_vm_pu"] -> bus_voltage_max_pu`
- `net.line["max_loading_percent"] -> line_loading_limit_percent`

fallback 只在字段缺失或非有限值时使用：

- bus min fallback: `0.95`
- bus max fallback: `1.05`
- line max fallback: `100.0`

AC OPF 和 DC OPF 都通过同一个 `_extract_opf_result()` 返回这些 limit 字段。若某次失败早于 pandapower net 可用阶段，limit dict 会保持为空，后续 metrics 仍会安全 fallback，不会崩溃。

## 5. voltage violation 新计算方式

`grid_model/grid_metrics.py` 的 `extract_grid_metrics()` 现在按 bus-specific limit 判断：

```text
voltage < bus_voltage_min_pu[bus] - tolerance
voltage > bus_voltage_max_pu[bus] + tolerance
```

默认 `tolerance = 1e-6`。

如果某个 bus limit 缺失，才 fallback 到函数参数中的 `voltage_min=0.95` / `voltage_max=1.05`。

正常电压偏离 `1.0` 不会计入 safe violation。

## 6. line overload 新计算方式

`grid_model/grid_metrics.py` 现在按 line-specific limit 判断：

```text
line_loading_percent > line_loading_limit_percent[line] + tolerance
```

如果某条 line limit 缺失，才 fallback 到 `line_loading_max=100.0`。

正常线路负载率不会计入 safe violation。

## 7. safe_violation_* 与 safe_cost_* 的关系

`env_wrappers/grid_coupled_env.py` 现在写入严格语义字段：

```python
safe_violation_voltage = float(grid_metrics.voltage_violation_count)
safe_violation_line = float(grid_metrics.line_overload_count)
safe_violation_opf = 0.0 if opf_result.success else 1.0
safe_violation_cost = safe_violation_voltage + safe_violation_line + safe_violation_opf
```

为了兼容旧代码，旧字段保留并等于新字段：

```python
safe_cost_voltage = safe_violation_voltage
safe_cost_line = safe_violation_line
safe_cost_opf = safe_violation_opf
safe_cost_total = safe_violation_cost
```

因此后续使用 `safe_cost_total` 的旧 eval / CSV 链路仍可工作，但它现在表示严格安全约束违规成本。

## 8. grid_security_penalty 新语义

`grid_security_penalty` 仍使用原权重形式：

```text
voltage_violation_count
+ 2.0 * line_overload_count
+ 100.0 * opf_infeasible_flag
```

但其中：

- `voltage_violation_count` 已改为 bus-specific limit violation count
- `line_overload_count` 已改为 line-specific loading limit violation count
- `opf_infeasible_flag = not opf_result.success`

所以 `grid_security_penalty` 不再由错误的全局 `0.95/1.05` 电压阈值驱动。

本次没有新增 `grid_pressure_index`。如果后续需要表达电网压力、质量或运行裕度，应单独设计，不应混入 `safe_violation_cost`。

## 9. eval 新增字段

`eval/eval_base.py` 新增 episode summary 指标：

- `total_safe_violation_cost`
- `total_safe_violation_voltage`
- `total_safe_violation_line`
- `total_safe_violation_opf`
- `avg_safe_violation_cost`
- `max_safe_violation_cost`

保留：

- `total_safe_cost`

当前 `total_safe_cost = total_safe_violation_cost`。

hourly row 新增：

- `safe_violation_voltage`
- `safe_violation_line`
- `safe_violation_opf`
- `safe_violation_cost`
- `grid_voltage_violation_magnitude`
- `grid_line_overload_magnitude`

旧的 hourly 字段 `safe_cost_voltage`、`safe_cost_line`、`safe_cost_opf`、`safe_cost_total` 保留。

eval 不重复统计 `grid_security_penalty`，`total_safe_cost` 由严格 `safe_violation_cost` 汇总。

## 10. smoke test 修改和结果

`scripts/smoke_test_grid_coupled_env.py` 现在每小时打印：

- `minV`
- `maxV`
- `maxLine%`
- `vCnt`
- `lineCnt`
- `safeV`
- `safeL`
- `safeOPF`
- `safeCost`
- `safeTotal`

summary 新增：

- `max_grid_voltage`
- `total_grid_security_penalty`
- `total_safe_violation_cost`
- `total_safe_violation_voltage`
- `total_safe_violation_line`
- `total_safe_violation_opf`
- `total_safe_cost`

运行结果摘要：

```text
final_obs_dim=264
action_dim=23
grid_scenario_enabled=True
grid_load_scale=0.910053
grid_reference_usep=160.840000
grid_opf_success_count=24
grid_opf_fail_count=0
grid_mef_success_count=24
grid_mef_fail_count=0
grid_opf_success_rate=1.000000
grid_mef_success_rate=1.000000
max_grid_line_loading=1.253392
min_grid_voltage=1.014641
max_grid_voltage=1.089860
final_grid_voltage_violation_count=0
final_grid_line_overload_count=0
total_grid_security_penalty=0.000000
total_safe_violation_cost=0.000000
total_safe_violation_voltage=0.000000
total_safe_violation_line=0.000000
total_safe_violation_opf=0.000000
total_safe_cost=0.000000
reward_mismatch_count=0
```

修复前后对比：

- 修复前 `total_safe_cost` 约 `194`
- 修复后当前 smoke test 下 `total_safe_cost = 0`

原因：当前 `maxV=1.089860` 没有超过该 bus 的 pandapower 上限，线路也没有超过 line-specific loading limit，OPF 全部成功。

## 11. 其他检查结果

编译检查通过：

```powershell
python -m py_compile grid_model/grid_case.py grid_model/opf_solver.py grid_model/grid_metrics.py env_wrappers/grid_coupled_env.py eval/eval_base.py scripts/smoke_test_grid_coupled_env.py
```

`scripts.smoke_test_grid_model` 通过：

- DC OPF success
- AC OPF success
- DC-MEF success
- AC-MEF success

`scripts.scan_ieee14_bus_sensitivity` 通过：

```text
OPF failed buses: None
```

该扫描刷新了节点敏感性 CSV/JSON 输出。

## 12. 本次未做事项

本次没有：

- 训练 PPO
- 修改 reward 权重
- 启用 grid reward
- 修改 `algorithms/`
- 修改服务器集群规模
- 修改 BESS 缩放逻辑
- 修改 NEMS 数据处理逻辑
- 使用 USEP 替代 OPF LMP
- 删除旧 safe cost 字段

## 13. 后续建议

1. 如果需要“电网运行压力”指标，建议单独设计 `grid_pressure_index` 或 `grid_voltage_quality_penalty`。
2. SAFE RL 后续应使用 `safe_violation_cost`，避免把普通运行压力混入安全约束违规。
3. grid reward ablation 可以在确认 `grid_security_penalty` 新语义后再小步推进。
4. 如果后续需要电压质量指标，可单独使用固定 deadband 或 deviation magnitude，但不要复用 `safe_cost_total`。

## 14. 最终结论

- safe_cost 是否已改为严格安全违规语义：是
- total_safe_cost 是否仍异常偏高：否，当前 smoke test 为 `0.000000`
- 当前 smoke test 下是否存在真实电压越限：否
- 当前 smoke test 下是否存在线路过载：否
- 当前 smoke test 下 OPF 是否全部成功：是，`24/24`
- 是否可以进入下一步 grid reward ablation / 小步 PPO smoke training：可以

