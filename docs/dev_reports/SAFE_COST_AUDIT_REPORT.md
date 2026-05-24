# SAFE_COST Audit Report

## 1. 检查文件列表

本次只做代码阅读、轻量运行和诊断报告，未修改业务代码。

检查的主要文件：

- `env_wrappers/grid_coupled_env.py`
- `grid_model/grid_metrics.py`
- `grid_model/grid_case.py`
- `grid_model/ieee14_loader.py`
- `grid_model/opf_solver.py`
- `eval/eval_base.py`
- `scripts/smoke_test_grid_coupled_env.py`
- `configs/config_ultimate.py`

额外全仓搜索：

- 搜索了 `safe_cost`、`grid_security_penalty`、`voltage_violation_count`、`line_overload_count`、`total_safe_cost` 等关键字。
- 除历史报告和数据输出外，当前业务计算链路集中在 `grid_metrics.py`、`grid_coupled_env.py`、`eval_base.py` 和 smoke test。

## 2. 当前 safe_cost_voltage 计算逻辑

位置：

- `grid_model/grid_metrics.py:10-31`
- `env_wrappers/grid_coupled_env.py:309`

当前逻辑：

```python
voltage_violation_count = sum(1 for value in voltages if value < voltage_min or value > voltage_max)
```

`extract_grid_metrics()` 的默认参数是：

```python
voltage_min = 0.95
voltage_max = 1.05
```

然后：

```python
safe_cost_voltage = float(grid_metrics.voltage_violation_count)
```

判断：

- 当前实现不是把正常电压偏离 `1.0` 也直接累计进去。
- 它只在电压低于 `0.95` 或高于 `1.05` 时计数。
- 但是该阈值是硬编码的全局阈值，不是 IEEE14 / pandapower OPF 的 bus-specific 电压约束。

重要发现：

- `grid_model/ieee14_loader.py` 会调用 `harmonize_generator_voltage_limits()`，使 IEEE14 的 bus 电压上限与发电机电压设定保持一致。
- 当前加载后的 pandapower bus 上限包括 `1.06`、`1.07`、`1.09`，下限是 `0.94`。
- 因此，OPF 可行且满足 pandapower bus 约束时，`grid_metrics.py` 仍可能因为全局 `1.05` 阈值将高电压计为 violation。

本次临时检查显示：

- smoke test 每小时 `maxV` 约为 `1.080430` 到 `1.089860`。
- 按 `grid_metrics.py` 的 `voltage_max=1.05`，每小时有 `7-9` 个电压 violation。
- 但按 pandapower 网络自身的 `min_vm_pu/max_vm_pu`，同一批小时的 bus-limit voltage violation count 为 `0`。

## 3. 当前 safe_cost_line 计算逻辑

位置：

- `grid_model/grid_metrics.py:14, 17, 24`
- `env_wrappers/grid_coupled_env.py:310`

当前逻辑：

```python
line_overload_count = sum(1 for value in line_loadings if value > line_loading_max)
```

`extract_grid_metrics()` 的默认参数是：

```python
line_loading_max = 100.0
```

然后：

```python
safe_cost_line = float(grid_metrics.line_overload_count)
```

判断：

- 当前实现只在线路 loading 超过 `100%` 时计数。
- 没有把正常线路负载率累计进 `safe_cost_line`。
- 当前 IEEE14 line 的 `max_loading_percent` 均为 `100.0`，所以此处与网络 line limit 一致。

本次 smoke test：

- `grid_max_line_loading_percent` 最大约 `1.253392%`。
- `grid_line_overload_count = 0`。
- `safe_cost_line = 0`。

## 4. 当前 safe_cost_opf 计算逻辑

位置：

- `env_wrappers/grid_coupled_env.py:311`

当前逻辑：

```python
safe_cost_opf = 0.0 if opf_result.success else 1.0
```

判断：

- OPF success 为 `True` 时，`safe_cost_opf` 一定为 `0.0`。
- OPF failed 时，`safe_cost_opf` 为 `1.0`。

本次 smoke test：

- OPF 成功率 `24/24`。
- `safe_cost_opf = 0`。

## 5. 当前 safe_cost_total 计算逻辑

位置：

- `env_wrappers/grid_coupled_env.py:309-312`

当前逻辑：

```python
safe_cost_voltage = float(grid_metrics.voltage_violation_count)
safe_cost_line = float(grid_metrics.line_overload_count)
safe_cost_opf = 0.0 if opf_result.success else 1.0
safe_cost_total = safe_cost_voltage + safe_cost_line + safe_cost_opf
```

结论：

- `safe_cost_total` 当前确实等于：

```text
safe_cost_voltage + safe_cost_line + safe_cost_opf
```

- 它不是直接等于 `grid_security_penalty`。
- 但在当前 smoke test 中，由于 `safe_cost_line=0`、`safe_cost_opf=0`，且 `grid_security_penalty=voltage_violation_count`，二者数值刚好相同。

## 6. 当前 grid_security_penalty 计算逻辑

位置：

- `grid_model/grid_metrics.py:23-31`
- `env_wrappers/grid_coupled_env.py:344`
- `env_wrappers/grid_coupled_env.py:362-374`

当前逻辑：

```python
grid_security_penalty = (
    float(voltage_violation_count)
    + 2.0 * float(line_overload_count)
    + (100.0 if opf_infeasible_flag else 0.0)
)
```

判断：

- 它当前不是一般线路负载率、网络压力或电压偏差指标。
- 它没有累计正常线路 loading。
- 它没有累计正常电压偏离 `1.0`。
- 它是基于 `voltage_violation_count`、`line_overload_count`、`opf_infeasible_flag` 的违规计数/权重指标。

问题：

- `voltage_violation_count` 使用硬编码 `0.95/1.05`，没有使用 OPF 网络自身的 bus 电压上下限。
- 因此 `grid_security_penalty` 形式上是违规成本，但在 IEEE14 当前设置下会把 OPF 可行、bus limit 内的高电压状态计为 violation。

额外说明：

- `grid_security_penalty` 会进入 grid observation，归一化参考值为 `grid_security_penalty_ref=10.0`。
- 如果 `enable_grid_reward=True`，它还会进入 `_compute_grid_reward_penalty()`。
- 当前 `configs/config_ultimate.py` 中 `enable_grid_reward=False`，`grid_security_weight=0.0`，所以 smoke test 中它不影响 reward。

## 7. eval 中 total_safe_cost 的汇总方式

位置：

- `eval/eval_base.py:236-280`
- `eval/eval_base.py:276-277`
- `eval/eval_base.py:341-493`

当前聚合逻辑：

```python
"total_grid_security_penalty": float(np.sum(_finite_values(episode_infos, "grid_security_penalty"))),
"total_safe_cost": float(np.sum(_finite_values(episode_infos, "safe_cost_total"))),
```

判断：

- `total_safe_cost` 是逐 step / 逐小时 `safe_cost_total` 求和。
- `total_grid_security_penalty` 单独求和。
- eval 没有把 `grid_security_penalty` 再叠加到 `total_safe_cost`。
- eval 没有重复统计 `grid_security_penalty`。
- eval 本身没有把正常线路负载或正常电压偏差计入 `total_safe_cost`；它只是汇总 env info 中已有的 `safe_cost_total`。

因此，eval 中 `total_safe_cost` 的语义完全依赖 `GridCoupledEnv` 中 `safe_cost_total` 的定义。

## 8. smoke test 中 total_safe_cost = 194 的可能来源

位置：

- `scripts/smoke_test_grid_coupled_env.py:95`
- `scripts/smoke_test_grid_coupled_env.py:130`
- `scripts/smoke_test_grid_coupled_env.py:172`

当前 smoke test 逻辑：

```python
total_safe_cost += _finite_or_zero(info.get("safe_cost_total"))
```

这与 eval 中逐小时求和 `safe_cost_total` 的定义一致。

本次运行命令：

```powershell
& 'C:\Users\bulio\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m scripts.smoke_test_grid_coupled_env
```

结果摘要：

- `grid_opf_success_count = 24`
- `grid_opf_fail_count = 0`
- `max_grid_line_loading = 1.253392`
- `min_grid_voltage = 1.014641`
- `total_safe_cost = 194.000000`

进一步临时字段拆分结果：

```text
TOTALS,grid_sec=194.000000,safe_v=194.000000,safe_line=0.000000,safe_opf=0.000000,safe_total=194.000000
```

逐小时来源：

- 每小时 `safe_cost_voltage` 为 `7-9`。
- 24 小时合计 `194`。
- 每小时 `safe_cost_line = 0`。
- 每小时 `safe_cost_opf = 0`。

关键原因：

- smoke test 只打印 `minV`，没有打印 `maxV`。
- 临时检查显示 `maxV` 约 `1.080430-1.089860`。
- 这超过 `grid_metrics.py` 的硬编码 `voltage_max=1.05`，所以被计入 `voltage_violation_count`。
- 但这些电压没有超过当前 pandapower IEEE14 case 的 bus-specific 上限，按网络自身 OPF 约束统计 violation count 为 `0`。

## 9. 当前 safe_cost 是否符合“只惩罚安全违规”的语义

结论：不完全符合。

更精确地说：

- 从代码结构看，`safe_cost` 的设计意图是只惩罚违规：电压越限、线路过载、OPF 失败。
- 它没有把正常线路负载率或正常电压偏离 `1.0` 直接计入。
- 但是当前电压越限判断使用硬编码 `0.95/1.05`，没有使用 OPF 模型实际采用的 bus-specific 约束。
- 因此，在 OPF 成功、线路不过载、且按 pandapower bus limits 没有电压越限时，仍可能得到非零 `safe_cost_voltage`。

按本项目当前 IEEE14 设置，`total_safe_cost=194` 很可能不是线路或 OPF 问题，而是电压安全阈值定义与 OPF 约束不一致导致的语义混淆。

## 10. 问题位置和原因

主要问题位置：

- `grid_model/grid_metrics.py:10-15`
- `grid_model/grid_metrics.py:23`
- `env_wrappers/grid_coupled_env.py:309`

原因：

1. `extract_grid_metrics()` 使用默认全局阈值：

```python
voltage_min=0.95
voltage_max=1.05
```

2. IEEE14 loader 中的实际 bus 电压约束不是统一 `0.95/1.05`：

```text
min_vm_pu = 0.94
max_vm_pu in {1.06, 1.07, 1.09}
```

3. OPF success 表示 pandapower OPF 已按其网络约束成功收敛，但 `grid_metrics.py` 又用更窄的全局阈值重新判定 violation。

4. smoke test 只打印了 `min_grid_voltage`，没有打印 `max_grid_voltage`，容易让人误以为电压完全没有触发当前代码里的 violation。

## 11. 建议修改方案，但本次不修改代码

### 方案 A：严格 safe cost

目标：

- `safe_cost` 只表示真实安全约束违规成本。
- OPF 成功、电压没有越限、线路没有过载时，`safe_cost_total=0`。

建议：

1. `safe_cost_voltage` 使用 OPF / pandapower 网络实际 bus limits：
   - 对每个 bus 使用 `min_vm_pu/max_vm_pu`。
   - 或在 `OPFResult` 中携带每个 bus 的电压上下限。
   - 或让 `extract_grid_metrics()` 接收 `GridCase/raw_result` 来读取 bus limits。

2. `safe_cost_line` 使用线路自身 `max_loading_percent`：
   - 当前 IEEE14 都是 `100.0`，但长期更稳妥。

3. `safe_cost_opf` 保持：

```text
0 if opf_success else positive penalty
```

4. `safe_cost_total` 保持三项求和，但三项均以真实安全约束为准。

优点：

- 最符合 safe RL 语义。
- `total_safe_cost` 可以直接解释为安全违规总成本。

风险：

- 历史结果中的 `total_safe_cost` 会发生变化。
- 如果下游已经把 `grid_security_penalty` / `safe_cost_total` 当成固定 `0.95/1.05` 电压质量指标使用，会改变含义。

### 方案 B：保留 grid_security_penalty 作为普通电网压力/质量指标，新增独立 safe_violation_cost

目标：

- 保留当前 `grid_security_penalty` 的可观测性和历史兼容性。
- 新增一个严格语义字段，例如：

```text
safe_violation_voltage
safe_violation_line
safe_violation_opf
safe_violation_cost
```

建议：

1. `grid_security_penalty` 可以继续作为 grid observation / reward shaping 的电网质量或压力指标。
2. 明确文档化：如果它使用固定 `0.95/1.05` 或其他质量阈值，它不是严格安全违规成本。
3. 新增 `safe_violation_cost`，只基于真实约束：
   - bus-specific voltage limit violation
   - line-specific loading limit violation
   - OPF failure
4. eval / smoke test 中新增或迁移：
   - `total_safe_violation_cost`
   - 保留 `total_grid_security_penalty`
   - 可选择后续将 `total_safe_cost` 指向严格 safe violation cost，或逐步弃用旧字段。

优点：

- 语义最清楚。
- 不把“电网压力/质量”与“安全违规成本”混在同一个字段。
- 对历史指标和下游依赖更友好。

风险：

- 指标数量增加。
- 需要同步更新 eval CSV、summary keys、报告解释和训练/评估文档。

## 12. 推荐方案

推荐方案：B。

理由：

- 当前项目已经有 `grid_security_penalty` 进入 grid observation / eval summary 的链路，直接改名或改变含义可能影响历史对比。
- 新增严格的 `safe_violation_cost` 能把 safe RL 约束语义独立出来。
- 后续如果确认不再需要旧含义，可以再把 `total_safe_cost` 迁移为严格 safe cost；但迁移前最好明确版本边界和指标命名。

如果希望最小改动并立刻修正 `total_safe_cost=194` 的解释问题，也可以采用方案 A；但从长期可维护性看，方案 B 更稳妥。

## 13. 最终结论

- 当前 safe_cost 是否存在语义混淆：是
- 当前 total_safe_cost=194 是否可能是 bug 或定义不合理：是
- 是否建议修改代码：是
- 推荐修改方案：B
- 修改前是否需要用户确认：是

