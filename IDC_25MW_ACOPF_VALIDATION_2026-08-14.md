# 25 MW IDC 现实尺度单点 AC OPF 验证

日期：2026-08-14

基线 HEAD：`016b319557a1f34c747a4dacd035362cd784552d`

范围：只比较当前约 1 MW IDC 与预先指定的 25 MW facility peak IDC。未训练 RL，未扫描其他 MW 尺度，未修改正式 IDC 规模、IEEE-14 参数、branch capacity、MARL/HGTA、reward、grid reward、Safe RL 或 Lagrangian。

## 1. 当前 IDC 尺度

当前功率模型的定义经代码确认是：

```text
P_IT = sum(server-group power) + 5 kW fixed distribution loss
P_cooling = P_IT / COP
P_IDC = P_IT + P_cooling + P_others
P_others = 2 kW
```

因此本报告中的“25 MW”是 facility-level total IDC demand，不是 IT power，也不是进入 OPF 前人为放大的 `P_grid`。

固定场景为 seed=2026、24h、23 维 action 全部为 0.5。最后一维 BESS action 经 `2a-1` 映射后为 0，故 BESS 在两个 case 中均不充不放。

当前配置保留 20 个 server groups，每组 100 台等效服务器，总计 2,000 台等效服务器；20 维 server action、异构组比例、调度逻辑和冷却模型均保持原状。

| 当前约 1 MW case | min MW | mean MW | max MW |
|---|---:|---:|---:|
| P_IT | 0.763603 | 0.763603 | 0.763603 |
| P_cooling | 0.241265 | 0.262925 | 0.286530 |
| P_others | 0.002000 | 0.002000 | 0.002000 |
| P_IDC | 1.006868 | 1.028528 | 1.052133 |
| bus 9 net load（PV/BESS 后） | 0.548577 | 0.870283 | 1.039296 |

P_IT 在这条固定 action/task 轨迹中恒定；24h facility 变化来自温度驱动的 COP/cooling 和 PV。这个现象是实际运行结果，不是把 P_IDC 写死。

## 2. 25 MW 校准方法

采用用户要求的约定：**固定策略 24h 的 facility peak demand≈25 MW**，不把 24h 平均功率锁死在 25 MW。

保持相同 server load 轨迹和单机异构比例时，group-size 只线性放大 server power/compute capacity；5 kW fixed loss 与 2 kW `P_others` 不随 fleet 放大。基准峰值发生在 hour 14，对应 COP=2.665。解析求解：

```text
target_IT = (25 MW - P_others) / (1 + 1/COP_peak)

group_size_exact =
    (target_IT - fixed_IT_loss)
    / ((baseline_peak_IT - fixed_IT_loss) / baseline_group_size)
```

结果：

| calibration item | value |
|---|---:|
| exact server_group_size | 2395.4911 |
| chosen integer server_group_size | 2395 |
| fleet scale vs current | 23.95× |
| predicted peak | 24.994876 MW |
| observed peak | 24.994876 MW |
| target error | −0.005124 MW |

这不是 MW 参数扫描：只从当前基准峰值物理分量解析反解目标整数 group size，然后运行一次目标 case 验证。

## 3. Fleet scale 与服务器等效数量变化

| item | current | 25 MW diagnostic | change |
|---|---:|---:|---:|
| server groups | 20 | 20 | 0 |
| server action dimensions | 20 | 20 | 0 |
| total action dimensions | 23 | 23 | 0 |
| equivalent servers/group | 100 | 2395 | 23.95× |
| effective server count | 2,000 | 47,900 | 23.95× |
| task workload scale | 100 | 2395 | 23.95× |

组内单机异构参数由相同 seed 生成，随后统一乘 group size；没有新增 graph/HGTA 节点，也没有改变 task scheduling 规则。

微电网严格保持当前实际尺度：

| microgrid item | unchanged value | relative to 24.9949 MW IDC peak |
|---|---:|---:|
| PV peak | 0.5 MW | 2.00% |
| BESS max charge/discharge | 2.0 MW | 8.00% |
| BESS energy | 10 MWh | peak-power equivalent duration 0.400 h |

这些比值仅说明现有 PV/BESS 相对 25 MW IDC 较小；本阶段未重新标定它们。

## 4. 25 MW 下 24h IDC 功率轨迹摘要

| 25 MW case | min MW | mean MW | max MW |
|---|---:|---:|---:|
| P_IT | 18.173538 | 18.173538 | 18.173538 |
| P_cooling | 5.742034 | 6.257545 | 6.819339 |
| P_others | 0.002000 | 0.002000 | 0.002000 |
| P_IDC | 23.917572 | 24.433083 | 24.994876 |
| bus 9 net load（PV/BESS 后） | 23.917572 | 24.274838 | 24.689351 |

PV 在白天最多抵消 0.5 MW；BESS neutral action 下功率为 0。完整逐小时轨迹见 `outputs/idc_25mw_acopf_validation/site_power_trajectories.csv`。

## 5. 约 1 MW vs 25 MW 对比

| metric | ~1 MW | 25 MW | change |
|---|---:|---:|---:|
| IDC facility peak MW | 1.052133 | 24.994876 | +23.942744 |
| bus 9 net peak MW | 1.039296 | 24.689351 | +23.650055 |
| AC OPF success | 24/24 | 24/24 | unchanged |
| global min voltage pu | 1.014573 | 1.016187 | +0.001613 |
| minimum lower margin pu | 0.074573 | 0.076187 | +0.001613 |
| minimum upper margin pu | ≈0 | ≈0 | regulated buses remain near upper setpoint |
| IDC bus minimum voltage pu | 1.053107 | 1.049846 | −0.003261 |
| max generator Q upper utilization | 82.03% | 89.98% | +7.95 percentage points |
| minimum generator Q upper headroom | 5.3917 Mvar | 3.0068 Mvar | −2.3849 Mvar |
| max generator P upper utilization | 40.95% | 45.13% | +4.18 percentage points |
| minimum generator P upper headroom | 59.0513 MW | 54.8748 MW | −4.1765 MW |
| mean network loss | 9.251215 MW | 9.414918 MW | +0.163703 MW (+1.77%) |
| peak ext-grid import | 197.053641 MW | 198.523066 MW | +1.469425 MW |
| max LMP spread | 4.629311 | 4.767734 | +0.138424 |
| mean IDC bus LMP | 40.125049 | 40.518430 | +0.393381 (+0.98%) |
| max line loading | 1.253075% | 1.257540% | +0.004466 percentage points |
| max transformer loading | 0.448046% | 0.531529% | +0.083483 percentage points |

全局最低电压略升并不表示 IDC 没有影响：OPF redispatch 改变了发电机电压/无功分配，而 bus 9 及其邻区电压明确下降。branch loading 只能作为 response metric；9900 MVA rating 未经物理重标定，不能据此宣称线路“特别安全”。

## 6. Voltage、generator P/Q 与 ext-grid headroom

### 6.1 逐 bus voltage

所有 margin 均使用各 bus 真实 `min_vm_pu/max_vm_pu`，不是统一 0.95–1.05。25 MW case 的 global critical lower-margin bus 仍为 pandapower bus 2（IEEE bus 3），minimum lower margin=0.076187 pu；无 violation。

| pp bus | IEEE bus | original min–max pu | 25 MW min–max pu | min-voltage change pu |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1.060000–1.060000 | 1.060000–1.060000 | ≈0 |
| 1 | 2 | 1.039740–1.040077 | 1.039785–1.040173 | +0.000046 |
| 2 | 3 | 1.014573–1.015787 | 1.016187–1.017049 | +0.001613 |
| 3 | 4 | 1.017932–1.019173 | 1.016923–1.017565 | −0.001010 |
| 4 | 5 | 1.019026–1.020799 | 1.018378–1.019285 | −0.000648 |
| 5 | 6 | 1.067998–1.069999 | 1.069995–1.070000 | +0.001998 |
| 6 | 7 | 1.059998–1.060000 | 1.057681–1.059999 | −0.002317 |
| 7 | 8 | 1.080182–1.089539 | 1.086208–1.090000 | +0.006025 |
| 8 | 9 / IDC | 1.053107–1.057758 | 1.049846–1.055636 | −0.003261 |
| 9 | 10 | 1.047558–1.053417 | 1.045058–1.051498 | −0.002499 |
| 10 | 11 | 1.053914–1.058571 | 1.053456–1.057416 | −0.000458 |
| 11 | 12 | 1.051378–1.057064 | 1.053127–1.057015 | +0.001749 |
| 12 | 13 | 1.046277–1.052870 | 1.047390–1.052408 | +0.001113 |
| 13 | 14 | 1.029997–1.039942 | 1.028588–1.038465 | −0.001409 |

### 6.2 25 MW source capability

`P/Q upper utilization=(value-min)/(max-min)`，用于判断向上供给能力；headroom 同时保留 lower/upper。对于无功短缺，Q upper headroom 比“离任意边界最近”更相关。

| source | bus | observed P MW | P bounds | min P upper headroom | max P util | observed Q Mvar | Q bounds | min Q upper headroom | max Q util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ext-grid 0 | 0 | 193.307–198.523 | 0–332.4 | 133.877 | 59.72% | 0.00009–0.00114 | 0–10 | 9.9989 | 0.01% |
| generator 0 | 1 | 36.496–37.537 | 0–140 | 102.463 | 26.81% | 15.339–18.963 | −40–50 | 31.037 | 65.51% |
| generator 1 | 2 | 20.345–45.125 | 0–100 | 54.875 | 45.13% | 19.609–24.596 | 0–40 | 15.404 | 61.49% |
| generator 2 | 5 | 0.000–5.320 | 0–100 | 94.680 | 5.32% | 8.988–17.043 | −6–24 | 6.957 | 76.81% |
| generator 3 | 7 | 10.140–36.070 | 0–100 | 63.930 | 36.07% | 16.240–20.993 | −6–24 | **3.007** | **89.98%** |

两个必要的限定：

1. ext-grid Q 几乎贴着其 0 Mvar **下界**，原始 case 也如此；严格按任意边界距离，它是数值上最近的约束。但它不是 25 MW 造成的“无功供应不足”，因为 Q 上裕度仍约 10 Mvar。
2. 对 25 MW 增量最敏感、且与无功供应压力相关的是 generator 3（pandapower bus 7 / IEEE bus 8）的 Q 上界：上裕度从 5.3917 降到 3.0068 Mvar。它尚未触界，但应作为后续监测重点。

P capability 很宽松：最小 generator P upper headroom 仍为 54.875 MW；ext-grid P upper headroom 为 133.877 MW。

## 7. AC OPF feasibility

- 25 MW case：24/24 success；failure hours：无；
- voltage violations：0；
- global voltage range：1.016187–1.090000 pu；
- minimum lower margin：0.076187 pu；
- minimum upper margin：约 0，因为调压母线合法地贴近自身上限；
- mean/max loss：9.414918/9.627427 MW；
- peak ext-grid import：198.523066 MW；
- max line/trafo response：1.257540% / 0.531529%；
- max IDC-bus LMP：40.757256；max system LMP spread：4.767734。

### 必答问题

**Q1. 25 MW 能否完成 24/24 AC OPF？**

能。24/24 成功，无 failure hour。

**Q2. 是否对 bus 9 及附近电网产生明显可辨识影响？**

是。bus 9 最低电压下降 0.003261 pu，bus 10 下降 0.002499 pu，局部 LMP、网损和 transformer response 均增加；影响可辨识但不剧烈。

**Q3. 相比约 1 MW，变化有多大？**

IDC bus mean LMP +0.3934，mean loss +0.1637 MW，IDC bus minimum voltage −0.003261 pu；generator 3 Q upper headroom −2.3849 Mvar。P headroom 变化较小，ext-grid peak import +1.4694 MW；其余新增负荷主要由 OPF 调整多台 generator dispatch 承担。

**Q4. 25 MW 是否接近当前物理安全/可行边界？**

结论：**moderately stressed in local/reactive response, but not near the overall feasibility boundary**。理由是 24/24、无 voltage violation、lower voltage margin 仍有 0.076 pu、P headroom 宽松；但某一发电机 Q upper utilization 已到约 90%，不能称为 very loose。

**Q5. 最先接近边界的物理因素是什么？**

对 25 MW 增量而言，是 generator 3 的 Q upper capability（剩余约 3.01 Mvar），其次才是局部 voltage response。generator P、ext-grid P 和 OPF feasibility 均不紧。ext-grid Q lower bound 从一开始就近似 active，需单独解释，不能误称为新增负荷导致的 reactive shortage。

## 8. 对 Safe RL 的含义

**Q7. 25 MW 下 Safe RL 是否已经具有潜在必要性？结论：Maybe。**

25 MW 已让逐 bus voltage、IDC LMP、loss 和 generator Q headroom 对 IDC 规模产生可学习的响应，因此未来加入基于真实 voltage/Q capability 的安全监测有物理动机。但本 case 没有 violation、OPF failure 或 P capability 紧张，单凭本次结果不能声称约束型 RL 已经必要，更不能恢复旧的 9900 MVA branch-loading 约束。

如果后续正式研究安全约束，优先级仍应是：

1. OPF feasibility；
2. bus-specific lower/upper voltage margin；
3. generator/ext-grid Q upper/lower headroom；
4. generator/ext-grid P headroom；
5. branch loading 仅作 response metric，直到 rating 重标定。

## 9. 是否建议 25 MW 作为后续正式主尺度

**Q6. 建议：Yes，作为现实尺度下的正式主尺度候选；本阶段不写入正式配置。**

理由不是“它最容易产生 Safe RL 效果”，而是：

- facility peak 校准到 24.995 MW，定义清晰；
- 相对 259 MW IEEE-14 原生负荷已足以形成可辨识的局部电压、LMP、loss 和 Q headroom 耦合；
- background=1.0 下仍为 24/24，可作为稳定主场景，而不是动辄 infeasible 的边界场景；
- 保持 20 groups 和现有动作/图结构，方法比较不被维度变化污染。

正式采用前仍建议单独确认：47,900 台等效服务器与论文所选服务器功率假设是否一致，以及 PV/BESS 是否要在另一个明确阶段按 25 MW 设施重新设计。本阶段没有做这些修改。

## 10. Tests

命令：

```text
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe -m pytest marl\tests\test_idc_25mw_acopf_probe.py -q
```

结果：`6 passed`。另有 pandapower 144 条 `tap_dependency_table` deprecation warnings，不影响数值结果。

| test | result | evidence |
|---|---|---|
| 正式 config unchanged | PASS | ENV/REWARD/DATA/IDC_SCALE/GRID/GRID_REWARD/GRID_SCENARIO 深拷贝前后相等 |
| seed reproducible | PASS | 两个 case 的 24h site trace 完全相等 |
| original ~1 MW baseline | PASS | peak=1.052133 MW，OPF=24/24，min V 与 Phase 2.5 一致 |
| 25 MW reproducible | PASS | group size=2395，peak=24.994876 MW，重复 OPF hourly/generator rows 完全相等 |
| 无 NaN/Inf | PASS | 递归 finite 检查；JSON `allow_nan=False`；缺失结果用 null |
| failure handling | PASS | 注入 synthetic failed OPF 后正确记录 hour/message，不使 diagnostic 崩溃；没有测试额外 MW 尺度 |

机器输出：

- `outputs/idc_25mw_acopf_validation/idc_25mw_acopf_validation.json`
- `outputs/idc_25mw_acopf_validation/site_power_trajectories.csv`
- `outputs/idc_25mw_acopf_validation/opf_hourly.csv`
- `outputs/idc_25mw_acopf_validation/generator_capability_hourly.csv`
- `outputs/idc_25mw_acopf_validation/bus_voltage_hourly.csv`
- `outputs/idc_25mw_acopf_validation/case_comparison.csv`

## 11. Git status

- HEAD before：`016b319557a1f34c747a4dacd035362cd784552d`
- 分支：`codex-model-improvement`
- 本阶段仅新增：`diagnostics/idc_25mw_acopf_probe.py`、`marl/tests/test_idc_25mw_acopf_probe.py`、本报告；机器结果位于 ignored `outputs/idc_25mw_acopf_validation/`。
- 正式配置和正式模型文件在运行前后未因本诊断发生变化。
- 仓库原有 dirty changes（`.gitignore`、`configs/config_ultimate.py`、`env_wrappers/grid_coupled_env.py`、`grid_model/grid_cache.py` 及既有 untracked 文件）均未修改、未暂存。

至此停止。未修改正式 IDC scale，未恢复 grid reward、Safe RL 或 Lagrangian，未测试 10 MW、100 MW 或其他 IDC 尺度。
