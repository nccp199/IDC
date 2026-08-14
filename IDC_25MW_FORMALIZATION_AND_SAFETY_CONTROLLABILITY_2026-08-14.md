# 25 MW IDC 正式化检查与 Safe RL 动作可控性评估

日期：2026-08-14

基准 HEAD：`016b319557a1f34c747a4dacd035362cd784552d`

分支：`codex-model-improvement`

本报告只审计 25 MW 主尺度的正式化条件，并在 IEEE-14、IDC 接入 bus 9、background multiplier=1.0、seed=2026 下测试 5 个固定动作场景。未训练 RL，未修改 MAPPO/HAPPO/HGTA、Safe RL、Lagrangian、reward、grid reward、branch capacity、IEEE-14 参数，也未测试其他 MW 尺度。

## 1. 结论摘要

**物理尺度定义通过，但当前训练配置不能直接正式化。**

- 最终应把“25 MW”定义为：最大合法 compute action 下、固定 24h 场景最高环境温度时的 facility-level rated demand，而不是 `action=0.5` 时的峰值。
- 按该定义，`server_group_size=1841`，20 组共代表 36,820 台等效服务器；额定 `P_IDC=25.001336 MW`。
- 旧 diagnostic 的 `server_group_size=2395` 是按 `action=0.5` 校准的；若采用最大合法 action，其额定功率会达到 `32.522146 MW`，不能作为正式 25 MW 定义。
- 任务工作量、容量、到达量、积压量等主体缩放链条基本保持比例一致；但发现 3 类必须先修复的训练尺度问题：SLA reward 被稀释、BESS degradation reward 被稀释、BESS/centralized state/HGTA 的原始功率特征尺度失配。
- 5 个动作场景全部 `24/24` AC OPF 成功，无电压越限。IDC 动作对 bus 9 电压与 generator 3 的 Q 裕度有明确控制作用；现有 BESS 有可辨识的瞬时作用，但受 10 MWh 容量限制，只持续 2–3 小时。
- 动作安全可控性分类：**B（moderate）**。IDC 动作作用明显，BESS 瞬时中等、全日持续性弱。
- 在本次固定 background=1.0 和合法极端动作范围内，Safe RL 结论为：**currently unnecessary**。最值得监测的是 generator Q 上界，但尚无实际 violation 或 OPF failure。

因此，本阶段没有修改正式 IDC scale。建议先完成第 4 节的最小尺度修复和回归测试，再把 1841 写入正式配置。

## 2. Action 语义与正式 25 MW 定义

### 2.1 当前动作语义

动作先被裁剪到 `[0,1]`。前 20 维是 server-group compute action：

```text
planned_task_load = server_action * 0.60
planned_total_load = base_load 0.05 + planned_task_load
```

因此：

| server action | planned task load | planned total load | 语义 |
|---:|---:|---:|---|
| 0.0 | 0.00 | 0.05 | 最低计算计划负载 |
| 0.5 | 0.30 | 0.35 | 中等计算计划负载 |
| 1.0 | 0.60 | 0.65 | 最大合法计算计划负载 |

实际负载仍由可用任务、完成工作量以及现有 reserve 逻辑决定，不是把功率写死。BESS action 使用 `2a-1` 映射：`0=最大充电`、`0.5=idle`、`1=最大放电`，并受功率、SOC 和不反送约束裁剪。

### 2.2 正式额定定义

最终定义为：

> **25 MW 是 `P_IDC=P_IT+P_cooling+P_others` 在最大合法 compute action（server action=1、planned total load=0.65）以及 seed=2026 固定 24h 场景最高温度 30°C 下的 facility-level rated demand。**

这一定义不要求每小时等于 25 MW，也不依赖某条固定 `action=0.5` 策略。

| 项目 | 数值 |
|---|---:|
| exact group size | 1840.901595 |
| chosen integer group size | **1841** |
| server groups | 20 |
| effective server count | **36,820** |
| scale vs current formal group size 100 | 18.41× |
| rated P_IT | 18.121940 MW |
| rated P_cooling | 6.877396 MW |
| P_others | 0.002000 MW |
| rated P_IDC | **25.001336 MW** |
| rated COP / PUE | 2.635 / 1.379617 |

作为定义审计而非额外 MW 扫描，旧 `group_size=2395` 在相同最大合法负载与温度下的额定值为 `32.522146 MW`。因此不能继续沿用 2395 并声称正式额定容量是 25 MW。

### 2.3 24h 功率范围

`group_size=1841` 的 site-only 参考轨迹：

| compute action | P_IT min/mean/max MW | P_cooling min/mean/max MW | P_IDC min/mean/max MW |
|---|---:|---:|---:|
| normal 0.5 | 13.970878 / 13.970878 / 13.970878 | 4.414179 / 4.810478 / 5.242356 | **18.387058 / 18.783356 / 19.215234** |
| high 1.0 | 15.248174 / 17.860954 / 18.121940（随实际任务供给） | 4.843934 / 6.218686 / 6.877396 | **20.094108 / 24.081640 / 25.001336** |

高动作前几小时实际任务供给不足，所以实际 P_IDC 低于额定值；峰值仍通过真实 server load → IT power → cooling power → facility power 链条达到 25.001336 MW。

## 3. 规模放大审计：A/B/C 分类

### A：自然按容量归一化，基本不受影响

- 单组计算容量、总计算容量和 sampled task workload 都随 `task_workload_scale/server_group_size` 成比例放大。
- 初始工作量、task arrival、remaining work、backlog、completed work、overdue/urgent work 等工作量变量按相同比例变化。
- `lambda_ref`、`queue_ref`、`queue_capacity_ref` 同比例放大；到达量、forecast、积压、完成工作量、unused capacity 等进入 observation/reward 前的比值保持一致。
- completion rate、task count、priority、deadline、waiting time 等本来就是比例、计数或时间变量，不应随设备数量机械放大。
- server capacity/efficiency 等 observation 使用相对或归一化表达；固定 seed 下异构比例和 20 个组节点保持不变。
- 测试确认 group size 100 与 1841 的 normalized true arrival 和 normalized noisy forecast 数组逐元素一致，forecast truth 隔离机制不变。

### B：比例改变，但当前数值仍可用

- 5 kW fixed IT loss 与 2 kW `P_others` 不随 fleet 放大；在 25 MW 尺度下影响极小，但论文中应明确它们是固定项。
- PV 仍为 0.5 MW peak，BESS 仍为 2 MW / 10 MWh；它们相对 IDC 的比例下降。这是本阶段有意保留的物理设定，不是数值错误。
- grid peak threshold/reference 随 IDC group scale 放大，而 PV/BESS 不放大；数值仍有限，但正式实验前应明确这是“IDC 额定比例阈值”而不是微电网资产同比例设计。
- clip 上限作用于已归一化 task 特征，未观察到因 18.41× 放大造成新增 clipping。

### C：正式训练前必须修复

1. **SLA reward 被稀释。** `sla_penalty` 是按任务优先级和延迟小时构成的计数/时间量，不随 workload 成比例增大，但当前 `sla_ref` 却乘以 `task_workload_scale`。从 group size 100 到 1841 后，相同 SLA 事件的归一化惩罚会再弱 18.41×。
2. **BESS degradation reward 被稀释。** 现有 BESS 保持 2 MW / 10 MWh，单步 degradation cost 没有随 IDC fleet 增大；但 reward 用随 IDC power scale 放大的 `cost_ref` 归一化，因此该惩罚相对再弱 18.41×。
3. **原始 supplemental power 特征尺度失配。** BESS actor 与 centralized state 直接拼接 `bess_energy_kWh, P_IDC_kW, P_grid_kW, charge_kW, discharge_kW`。正式 HARL 的 `use_feature_normalization=true` 实际是单样本 `LayerNorm(obs_dim)`，不是逐特征 running normalization，不能保证从约 1,000 kW 跳到约 25,000 kW 后语义保持一致。
4. **HGTA 固定 reference 失配。** 当前 HGTA 使用 `idc_power_ref_kw=2000`、`grid_power_ref_kw=4000`；25 MW 下 IDC node 输入约为 12.5，而原设定接近 1。只改 YAML reference 仍不能解决 BESS actor 和 MLP critic 的原始 supplemental 特征问题。

结论：存在 C 类问题，所以“直接把正式 group size 改成 1841”不通过本次正式化 gate。

## 4. 建议的最小正式化修复

不改变 actor/action 维度、20 个 server groups、HGTA 节点数量或算法结构，仅建议：

1. 在 formal config 中显式记录 `facility_rated_power_mw=25.0`、`server_group_size=1841`、`task_workload_scale=1841`，并注明额定条件是 action=1、load=0.65、30°C。
2. `sla_ref` 不再乘 workload scale，或另设 count/time-based `sla_penalty_ref`；保持 SLA 事件相对权重不因 fleet 变化。
3. 新增独立 `bess_degradation_cost_ref`，按当前 BESS 的功率/退化成本定义，不复用 IDC-scaled `cost_ref`。
4. 为 6 个 supplemental features 建立明确固定 references：SOC、10,000 kWh、25,000 kW IDC、约 27,000 kW grid、2,000 kW charge、2,000 kW discharge；在 BESS observation 与 centralized critic 输入处一致地归一化一次。HGTA 对这些已归一化位置不得再次除 reference。
5. 增加同维度回归测试，并把 feature-semantics 变更写入 checkpoint metadata；旧 checkpoint 即使维度相同，也不应默认为输入语义兼容。

这是小范围尺度修复，不需要改 MAPPO/HAPPO/HGTA 结构。完成后才建议把 1841 正式写入训练配置。

## 5. 五个动作场景与 AC OPF 结果

所有场景共享 seed=2026、同一 task/PV/temperature/price/grid profile、IEEE-14、bus 9 和 background=1.0，仅改变合法动作。BESS 为当前 2 MW / 10 MWh、初始 SOC=0.5、范围 0.1–0.9。

| case | server action | BESS action | P_IDC min/mean/max MW | bus 9 net peak MW | BESS 实际作用 | OPF |
|---|---:|---|---:|---:|---|---:|
| low IDC + discharge | 0 | max discharge | 11.348394 / 11.588629 / 11.850201 | 11.708067 | 2 MW peak，3.8 MWh，2 h | 24/24 |
| low IDC + charge | 0 | max charge | 11.348394 / 11.588629 / 11.850201 | 13.377048 | 2 MW peak，4.2105 MWh，3 h | 24/24 |
| high IDC + idle | 1 | idle | 20.094108 / 24.081640 / 25.001336 | 24.689860 | 0 | 24/24 |
| high IDC + charge | 1 | max charge | 20.094108 / 24.081640 / 25.001336 | 25.966887 | 2 MW peak，4.2105 MWh，3 h | 24/24 |
| high IDC + discharge | 1 | max discharge | 20.094108 / 24.081640 / 25.001336 | 24.689860 | 2 MW peak，3.8 MWh，2 h | 24/24 |

所有 120 次 AC OPF 均成功，failure hour 为空，无 NaN/Inf。充电能量与放电能量不同来自 95% charge/discharge efficiency 和 SOC 上下限，不是动作未生效。

## 6. Voltage、source capability 与 grid response

### 6.1 场景摘要

| metric | low discharge | low charge | high idle | high charge | high discharge |
|---|---:|---:|---:|---:|---:|
| global min V pu | 1.015666 | 1.015666 | 1.016067 | 1.016076 | 1.016041 |
| global max V pu | 1.089999 | 1.089999 | 1.090000 | 1.090000 | 1.090000 |
| minimum lower margin pu | 0.075666 | 0.075666 | 0.076067 | 0.076076 | 0.076041 |
| minimum upper margin pu | -6.2e-11 | -6.2e-11 | -5.6e-11 | -5.6e-11 | -5.6e-11 |
| IDC bus min V pu | 1.052307 | 1.052307 | 1.049846 | 1.049846 | 1.049846 |
| IDC bus mean V pu | 1.054659 | 1.054634 | 1.053272 | 1.053258 | 1.053284 |
| voltage violations | 0 | 0 | 0 | 0 | 0 |
| max source Q upper utilization | 84.99% | 84.99% | 89.98% | 89.98% | 89.98% |
| min source Q upper headroom | 4.5019 | 4.5019 | 3.0071 | 3.0071 | 3.0071 Mvar |
| max source P upper utilization | 59.47% | 59.47% | 59.72% | 59.72% | 59.72% |
| min source P upper headroom | 57.1813 | 57.1813 | 54.8751 | 54.8751 | 54.8751 MW |
| mean network loss MW | 9.324927 | 9.327142 | 9.412203 | 9.413563 | 9.411004 |
| peak ext-grid import MW | 197.6813 | 197.6813 | 198.5229 | 198.5229 | 198.5229 |
| mean IDC-bus LMP | 40.31779 | 40.32284 | 40.51311 | 40.51577 | 40.51071 |
| max LMP spread | 4.69712 | 4.69712 | 4.76772 | 4.76772 | 4.76772 |
| max line response | 1.25467% | 1.25467% | 1.25754% | 1.25754% | 1.25754% |
| max transformer response | 0.45754% | 0.45754% | 0.53152% | 0.53152% | 0.53152% |

所有 voltage margin 都使用每个 bus 自己的 `min_vm_pu/max_vm_pu`。critical lower-margin bus 均为 pandapower bus 2（IEEE bus 3）。约 `-6e-11` 的 upper margin 是调压母线贴近自身上限时的求解器数值容差，不构成物理越限。全局最低电压随 OPF redispatch 略升，并不否定局部影响；更直接的指标是 IDC bus 电压从 low 到 high 下降 `0.002461 pu`。

### 6.2 Generator 与 ext-grid

高 IDC idle 的关键 source capability：

| source | P observed MW | min P upper headroom | Q observed Mvar | Q bounds | min Q upper headroom |
|---|---:|---:|---:|---:|---:|
| ext-grid 0 | 193.233–198.523 | 133.877 MW | 0.00009–0.00114 | 0–10 | 9.9989 Mvar |
| generator 0 | 36.482–37.536 | 102.464 MW | 15.314–18.963 | -40–50 | 31.0374 Mvar |
| generator 1 | 20.150–45.125 | 54.875 MW | 19.625–24.596 | 0–40 | 15.4040 Mvar |
| generator 2 | 0.00014–5.319 | 94.681 MW | 8.987–17.043 | -6–24 | 6.9571 Mvar |
| generator 3 | 8.865–36.069 | 63.931 MW | 15.889–20.993 | -6–24 | **3.0071 Mvar** |

ext-grid Q 数值接近其 0 Mvar 下界，但上裕度约 10 Mvar；这不是新增负荷造成的 Q shortage。对 25 MW 增量最敏感、且最先接近相关上界的是 generator 3 的 Q capability，最大 Q upper utilization 为 89.98%。P capability 与 ext-grid P/Q upper capability 都较宽松。

branch rating 仍是约 9900 MVA，line/transformer loading 只能作为 response metric，不能据此宣称 thermal safety。

## 7. 动作安全可控性

### 7.1 IDC compute action

在相同 BESS 动作下，从 low IDC 切换到 high IDC：

- bus 9 net peak 增加 `12.59–12.98 MW`；
- IDC bus minimum voltage 下降 `0.002461 pu`，mean voltage 下降约 `0.001375 pu`；
- generator 3 的最小 Q upper headroom 从 `4.5019` 降至 `3.0071 Mvar`，减少 `1.4948 Mvar`；
- source 最小 P upper headroom 减少 `2.3062 MW`；
- mean network loss 增加约 `0.086 MW`；
- mean IDC-bus LMP 增加约 `0.193`，max LMP spread 增加 `0.0706`。

因此 IDC compute action 对相关安全裕度具有明确、可辨识、方向一致的控制作用，但本次最大合法 action 仍未触界。

### 7.2 BESS action

高 IDC 下，对同一小时的“最大充电减最大放电”：

| paired hourly effect | mean | maximum absolute effect |
|---|---:|---:|
| bus 9 net load | +0.3338 MW | **+4.0000 MW** |
| IDC bus voltage | -0.000026 pu | **-0.000319 pu** |
| network loss | +0.002559 MW | **+0.031367 MW** |
| ext-grid import | +0.027753 MW | **+0.332523 MW** |
| IDC-bus LMP | +0.005055 | **+0.060614** |
| generator 3 Q | +0.0532 Mvar | **+0.660354 Mvar** |
| generator 3 Q upper headroom | -0.0532 Mvar | **-0.660354 Mvar** |

放电相对充电会提高 bus 9 电压、增加 generator 3 Q 上裕度、降低 loss 与 LMP，方向符合物理预期；但初始可用 SOC 只支持最大放电 2 小时，最大充电只持续 3 小时，所以最差 24h summary 往往发生在 BESS 已饱和后，5 个 case 的 OPF feasibility 和全日关键极值没有改变。

综合分类：**B（moderate controllability）**。IDC action 明显；BESS 瞬时有效，但在当前 10 MWh 下缺乏全天持续控制能力，不能评为 A。

## 8. 必答问题与 Safe RL 含义

**Q1. 25 MW 能否正式化？**

物理额定定义可以；当前训练配置不能直接正式化。C 类 reward/normalization 问题修复并回归通过后，建议用 `group_size=1841` 正式化。

**Q2. 最终正式定义是什么？**

不是 `action=0.5 peak=25 MW`，而是最大合法 server action=1、planned total load=0.65、固定场景最高温度 30°C 时，facility-level `P_IT+P_cooling+P_others≈25 MW`。

**Q3. 放大后是否存在 task/reward/observation 尺度问题？**

task work/capacity 主链条基本自然归一化；必须修复 SLA reward、BESS degradation reward、raw supplemental features 和 HGTA fixed references。详见第 3–4 节。

**Q4. IDC 动作能否控制安全裕度？**

能。low→high 使 bus 9 minimum voltage 下降 0.002461 pu，并使关键 generator 3 Q upper headroom 减少 1.4948 Mvar；作用可辨识，但仍未造成 violation。

**Q5. BESS 是否改善/恶化 voltage、Q 和 feasibility？**

最大放电相对最大充电在有效小时内改善 bus 9 voltage 最多 0.000319 pu、改善 generator 3 Q upper headroom 最多 0.660 Mvar；最大充电反向恶化。现有容量不足以改变 24/24 feasibility 或全日最差极值。

**Q6. Safe RL 是否已经具有潜在必要性？**

结论：**NO / currently unnecessary for the tested background=1.0 scenario**。理由是 5 个合法代表性极端动作场景共 120 次 OPF 全部成功、无 voltage violation，P/ext-grid 裕度宽松；虽然 generator 3 Q 利用率约 90%，但仍有约 3.01 Mvar 上裕度。不能为了 Safe RL 预设需求而把“可监测”解释成“必须约束”。

如未来场景出现 violation，约束优先级应为：

1. generator/ext-grid Q upper/lower headroom；
2. per-bus voltage margin；
3. OPF feasibility；
4. generator/ext-grid P headroom；
5. branch loading 在真实 rating 重标定前不作为 hard safety constraint。

## 9. Tests 与可复现性

执行命令：

```text
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe -m pytest marl\tests\test_idc_25mw_safety_controllability.py marl\tests\test_task_forecast_isolation.py -q
```

结果：`11 passed, 3 skipped`。3 个 skip 是原 forecast suite 中依赖未加入当前 Python path 的外部 HARL package 的可选测试，不是失败；本任务自己的 5 个测试全部执行并通过。

提交前又将上一阶段的 `test_idc_25mw_acopf_probe.py` 一并回归，最终结果为 `17 passed, 3 skipped`；另有 144 条 pandapower 3.x `tap_dependency_table` deprecation warning，不影响数值结果。

| 检查 | 结果 |
|---|---|
| 正式 config 运行前后不变 | PASS |
| rated group size / 25 MW 定义 | PASS：1841 / 25.001336 MW |
| actor/action shape | PASS：flat 23，20 server actions 不变 |
| 5 个 site action case seed replay | PASS：逐条完全相等 |
| 完整 5-case AC OPF 重放 | PASS：JSON SHA-256 两次均为 `9B4CEA59C5F8F50850E578A2B76191035F30BD64F9731CBEE3262097410B00E7` |
| forecast 机制与 normalized 值不变 | PASS |
| 无 NaN/Inf | PASS；JSON 使用 `allow_nan=False` |
| 5 个 AC OPF case | PASS：各 24/24 |
| synthetic OPF failure 不崩溃 | PASS：正确记录 failure hour/message |

Diagnostic：`diagnostics/idc_25mw_safety_controllability.py`

机器输出：`outputs/idc_25mw_safety_controllability/`（ignored，不提交）

## 10. Git status 与提交边界

- 本任务开始时 HEAD：`016b319557a1f34c747a4dacd035362cd784552d`。
- 本任务没有修改正式配置、MARL/HGTA、reward、grid reward、Safe RL 或电网参数。
- 仓库原有 tracked dirty changes：`.gitignore`、`configs/config_ultimate.py`、`env_wrappers/grid_coupled_env.py`、`grid_model/grid_cache.py`；这些不是本任务产生的，保持未暂存。
- 仓库还存在多份原有 untracked 文档与测试；除本次及其直接依赖的 25 MW diagnostic 文件外，均不纳入提交。
- 本次提交只包含 25 MW diagnostic 脚本、相应测试和两份 25 MW 报告；生成的 `outputs/` 不提交。

至此停止。没有写入正式 25 MW scale，没有恢复 grid reward/Safe RL/Lagrangian，没有测试其他 MW 尺度。
