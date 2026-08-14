# Phase 2.5：IEEE-14 电网容量审计与压力区间扫描

日期：2026-08-14

仓库基线：`e777d841e21892b97971486129d1095b7d3d149c`

测试环境：`idc_ppo`，pandapower 3.4.0，AC OPF
结论状态：完成；未修改正式训练默认负荷、线路/变压器容量、MARL、HGTA、actor/critic、centralized state、forecast、reward、Safe RL、BESS 或任务模型。

## 1. Capacity audit

### 1.1 网络来源与项目修改

`grid_model/ieee14_loader.py` 直接加载 `pandapower.networks.case14()`，系统基准容量为 100 MVA。项目随后只执行 `harmonize_generator_voltage_limits()`：若 generator voltage setpoint 超出母线边界，就把对应母线边界扩展到该 setpoint；它不修改拓扑、线路、变压器、负荷或发电机位置。

相对 pandapower 原始 case14，实际加载后的差异只有：

- pandapower bus 5（IEEE bus 6）`max_vm_pu: 1.06 -> 1.07`；
- pandapower bus 7（IEEE bus 8）`max_vm_pu: 1.06 -> 1.09`。

IDC 接在 pandapower bus 8，即 IEEE bus 9。

### 1.2 Bus 与电压边界

所有 `min_vm_pu`、`max_vm_pu` 都是有限显式值，不是本项目求解器补入的 NaN fallback。母线类型根据 ext-grid/gen/load 元件推断；`net.bus.type` 自身只是通用 `b`。

| pp bus | IEEE bus | vn_kV | min | max | 角色 | 基准 24h observed min–max | 最小 lower margin | 最小 upper margin | violation |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 1 | 135 | 0.94 | 1.06 | Slack / ext-grid | 1.060000–1.060000 | 0.120000 | ≈0 | 0 |
| 1 | 2 | 135 | 0.94 | 1.06 | PV / gen | 1.039740–1.040077 | 0.099740 | 0.019923 | 0 |
| 2 | 3 | 135 | 0.94 | 1.06 | PV / gen | 1.014573–1.015787 | 0.074573 | 0.044213 | 0 |
| 3 | 4 | 135 | 0.94 | 1.06 | PQ | 1.017932–1.019173 | 0.077932 | 0.040827 | 0 |
| 4 | 5 | 135 | 0.94 | 1.06 | PQ | 1.019026–1.020799 | 0.079026 | 0.039201 | 0 |
| 5 | 6 | 0.208 | 0.94 | 1.07 | PV / gen | 1.067998–1.069999 | 0.127998 | 0.000001 | 0 |
| 6 | 7 | 14 | 0.94 | 1.06 | 无注入连接母线 | 1.059998–1.060000 | 0.119998 | ≈0 | 0 |
| 7 | 8 | 12 | 0.94 | 1.09 | PV / gen | 1.080182–1.089539 | 0.140182 | 0.000461 | 0 |
| 8 | 9 | 0.208 | 0.94 | 1.06 | PQ / IDC | 1.053107–1.057758 | 0.113107 | 0.002242 | 0 |
| 9 | 10 | 0.208 | 0.94 | 1.06 | PQ | 1.047558–1.053417 | 0.107558 | 0.006583 | 0 |
| 10 | 11 | 0.208 | 0.94 | 1.06 | PQ | 1.053914–1.058571 | 0.113914 | 0.001429 | 0 |
| 11 | 12 | 0.208 | 0.94 | 1.06 | PQ | 1.051378–1.057064 | 0.111378 | 0.002936 | 0 |
| 12 | 13 | 0.208 | 0.94 | 1.06 | PQ | 1.046277–1.052870 | 0.106277 | 0.007130 | 0 |
| 13 | 14 | 0.208 | 0.94 | 1.06 | PQ | 1.029997–1.039942 | 0.089997 | 0.020058 | 0 |

定义严格按任务要求：`lower_margin = vm_pu - min_vm_pu`，`upper_margin = max_vm_pu - vm_pu`。如果把两侧 margin 合并取最小，基准值约为 0，因为 slack/PV 调压母线合法地贴近其上边界；这不是负荷导致的欠压风险。评估负荷压力时，应同时报告两侧 margin，并重点观察 minimum lower margin。

### 1.3 Line 容量

pandapower 当前实现的线路 loading denominator 是：

```text
i_actual = max(i_from_ka, i_to_ka)
i_permitted = max_i_ka * df * parallel
loading_percent = 100 * i_actual / i_permitted
```

下表的近似 thermal MVA 仅为 `sqrt(3) * vn_from_kV * i_permitted`，不是精确 MW 极限；功率因数、端电压与损耗会影响实际 MW。

| line | from–to | length km | r Ω/km | x Ω/km | c nF/km | max_i kA | df | parallel | max load % | approx MVA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0–1 | 1 | 3.532005 | 10.783733 | 768.4848 | 42.3390 | 1 | 1 | 100 | 9900 |
| 1 | 0–4 | 1 | 9.846968 | 40.649040 | 716.0881 | 42.3390 | 1 | 1 | 100 | 9900 |
| 2 | 1–2 | 1 | 8.563928 | 36.080033 | 637.4931 | 42.3390 | 1 | 1 | 100 | 9900 |
| 3 | 1–3 | 1 | 10.590548 | 32.134320 | 494.8576 | 42.3390 | 1 | 1 | 100 | 9900 |
| 4 | 1–4 | 1 | 10.379138 | 31.689630 | 503.5904 | 42.3390 | 1 | 1 | 100 | 9900 |
| 5 | 2–3 | 1 | 12.212573 | 31.170218 | 186.2993 | 42.3390 | 1 | 1 | 100 | 9900 |
| 6 | 3–4 | 1 | 2.433038 | 7.674548 | 0 | 42.3390 | 1 | 1 | 100 | 9900 |
| 7 | 5–10 | 1 | 0.000041 | 0.000086 | 0 | 27479.6522 | 1 | 1 | 100 | 9900 |
| 8 | 5–11 | 1 | 0.000053 | 0.000111 | 0 | 27479.6522 | 1 | 1 | 100 | 9900 |
| 9 | 5–12 | 1 | 0.000029 | 0.000056 | 0 | 27479.6522 | 1 | 1 | 100 | 9900 |
| 10 | 8–9 | 1 | 0.000014 | 0.000037 | 0 | 27479.6522 | 1 | 1 | 100 | 9900 |
| 11 | 8–13 | 1 | 0.000055 | 0.000117 | 0 | 27479.6522 | 1 | 1 | 100 | 9900 |
| 12 | 9–10 | 1 | 0.000035 | 0.000083 | 0 | 27479.6522 | 1 | 1 | 100 | 9900 |
| 13 | 11–12 | 1 | 0.000096 | 0.000086 | 0 | 27479.6522 | 1 | 1 | 100 | 9900 |
| 14 | 12–13 | 1 | 0.000074 | 0.000151 | 0 | 27479.6522 | 1 | 1 | 100 | 9900 |

全部 `std_type=None`、`length_km=1`、`df=1`、`parallel=1`、`max_loading_percent=100`。高压线路允许电流为 42.339 kA；0.208 kV 侧线路为 27479.652 kA。二者换算后都精确落到约 9900 MVA。这是 case14/MATPOWER 转换保留的统一巨大 `RATE_A`，不是逐线路工程额定值。

### 1.4 Transformer 容量

runopp 的 `trafo_loading='current'`。当前 denominator 等价于每侧电流折算的视在功率除以 `sn_mva * df * parallel` 后取较大侧；OPF thermal rate 为 `sn_mva * df * parallel * max_loading_percent / 100`。

| trafo | hv–lv bus | sn MVA | hv/lv kV | vk % | vkr % | parallel | df | max load % | permitted MVA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3–6 | 9900 | 135/14 | 2070.288 | 0 | 1 | 1 | 100 | 9900 |
| 1 | 3–8 | 9900 | 135/0.208 | 5506.182 | 0 | 1 | 1 | 100 | 9900 |
| 2 | 4–5 | 9900 | 135/0.208 | 2494.998 | 0 | 1 | 1 | 100 | 9900 |
| 3 | 6–7 | 9900 | 14/12 | 1743.885 | 0 | 1 | 1 | 100 | 9900 |
| 4 | 6–8 | 9900 | 14/0.208 | 1089.099 | 0 | 1 | 1 | 100 | 9900 |

异常大的 `sn_mva` 与 `vk_percent` 是以统一 9900 MVA 基准保存原 p.u. 阻抗的转换表示，不是五台真实设备的铭牌容量。

## 2. OPF constraint audit

### 2.1 当前 AC OPF 实际 enforce 的约束

| 约束 | metric 是否存在 | OPF 是否 enforce | 证据与结论 |
|---|---|---|---|
| Bus voltage VMIN/VMAX | 是 | 是 | 每个 bus 的 `min_vm_pu/max_vm_pu` 进入内部 ppc；成功解都在各自边界内。不能用统一 0.95–1.05 替代。 |
| Generator P/Q | 是 | 是 | `runopp` options 显示 `enforce_p_lims=True`、`enforce_q_lims=True`；gen/ext-grid 上下界写入 ppc。 |
| Ext-grid P/Q | 是 | 是 | bus 0：P 0–332.4 MW，Q 0–10 Mvar；作为可控注入参与 OPF。 |
| Line thermal | 是 | 是，但不具辨识度 | 15 条 line 的 `RATE_A≈9900 MVA`；私人副本把全部 limit 降为 0.001% 后 OPF 不收敛，证明不是只计算 metric。 |
| Transformer loading | 是 | 是，但不具辨识度 | 5 台 trafo 同样进入内部 branch `RATE_A=9900 MVA`；当前 `trafo_loading=current`。 |

内部 ppc 的 20 个 branch `RATE_A` 全部为 9899.999999999998 或 9900.0 MVA。必须区分：

- `loading_percent` 能被计算；
- branch `RATE_A` 确实被 OPF enforce；
- 但 9900 MVA 是极宽松的占位式额定值，因此约束在本实验负荷域内几乎不可能成为 active constraint。

## 3. Normal scenario diagnosis

### 3.1 24h 基准复现

固定 seed 2026、固定 23 维 action=0.5，得到：

- AC OPF：24/24；
- 原生背景负荷：259 MW + 73.5 Mvar，时变 scale 0.876997–1.113496；
- 物理 IDC demand：1.0069–1.0521 MW；PV/BESS 平衡后 IDC bus 净购电：0.5486–1.0393 MW；
- bus voltage：1.014573–1.089539 pu；
- max line loading：1.253075%；
- max transformer loading：0.448046%；
- mean network loss：9.2512 MW；peak ext-grid import：197.0536 MW；
- LMP spread max：4.6293；
- configured voltage/line/trafo violation：全部 0。

### 3.2 每条支路基准统计

| line | max % | mean % | p95 % | line | max % | mean % | p95 % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.2531 | 1.2345 | 1.2524 | 8 | 0.0843 | 0.0762 | 0.0839 |
| 1 | 0.6302 | 0.6217 | 0.6301 | 9 | 0.1934 | 0.1764 | 0.1925 |
| 2 | 0.5601 | 0.5400 | 0.5585 | 10 | 0.0863 | 0.0716 | 0.0860 |
| 3 | 0.4787 | 0.4754 | 0.4786 | 11 | 0.1195 | 0.1026 | 0.1191 |
| 4 | 0.3663 | 0.3609 | 0.3660 | 12 | 0.0360 | 0.0327 | 0.0359 |
| 5 | 0.1550 | 0.1210 | 0.1521 | 13 | 0.0175 | 0.0162 | 0.0174 |
| 6 | 0.5161 | 0.5078 | 0.5160 | 14 | 0.0529 | 0.0512 | 0.0526 |
| 7 | 0.0716 | 0.0695 | 0.0712 |  |  |  |  |

| trafo | max % | mean % | p95 % |
|---:|---:|---:|---:|
| 0 | 0.2733 | 0.2419 | 0.2727 |
| 1 | 0.1503 | 0.1475 | 0.1497 |
| 2 | 0.4480 | 0.4318 | 0.4465 |
| 3 | 0.2603 | 0.1765 | 0.2564 |
| 4 | 0.3815 | 0.3100 | 0.3790 |

最高线路的近似实际视在量级为 `1.253075% × 9900 ≈ 124.05 MVA`，并不算“几乎没有潮流”；最高变压器约为 `0.448046% × 9900 ≈ 44.36 MVA`。低百分比主要由 9900 MVA denominator 造成。

### 3.3 Q1：为什么最高 line loading 只有约 1.25%（按重要性排序）

1. **主因：全部线路的 configured capacity 都约为 9900 MVA。** 它比当前百 MW 级潮流大约两个数量级。
2. **case14 数据没有提供可用于本项目安全研究的逐线路工程 thermal rating。** `max_i_ka` 是从统一 9900 MVA rate 反算出来的，低压侧尤其出现 27479.65 kA，不应解释为真实导线载流量。
3. **当前场景规模也较温和。** 原生负荷为 259 MW，IDC bus 净增量仅约 0.55–1.04 MW；但“潮流小”是次因，因为最忙线路仍承载约 124 MVA。
4. **项目代码没有人为放大 branch capacity。** loader 对 case14 唯一修改是两处 generator voltage upper bound harmonization；未修改 line/trafo rating。
5. **Transformer 有相同问题。** 五台统一 `sn_mva=9900`，所以约 44 MVA 也只显示 0.45%。

## 4. Stress sweep configuration

### 4.1 固定物理轨迹

- seed：2026；24h；不训练 RL；
- task scenario、forecast、PV、temperature、price 均固定为现有 `main` case 和同一 seed；
- 23 维 action 全部为 0.5：20 个 server action、urgent preference、continuity preference 均固定；最后 BESS action 经现有映射 `2a-1` 后为 0，故 BESS 物理充放电为 0；
- 先由真实 IDC/BESS/PV transition 生成 24h 物理轨迹，再关闭该私人 wrapper 实例的网格求解，避免重复 OPF/MEF 干扰；
- 每个 stress 点都用 `solve_ac_opf` 的深拷贝网络独立求解。

### 4.2 两个倍率的物理含义

- Background multiplier 作用于 case14 原生 load table 的 P、Q scaling，并叠乘现有 24h `grid_load_scale_t`；
- IDC multiplier 作用于**已经经过 IDC demand、BESS、PV 能量平衡后实际得到的 IDC bus 净购电 MW**。它没有改写 IDC 内部模型，也没有分别放大 BESS/PV，从而不会伪造内部能量平衡；它表示同构站点/等比例站点净接入的 physics-only probe。

扫描集合：

- background axis：1, 1.25, 1.5, 1.75, 2, 2.5, 3（IDC=1）；
- IDC axis：1, 1.5, 2, 2.5, 3, 4（background=1）；
- coarse 2D：background {1,1.5,2,2.5} × IDC {1,1.5,2,3}；
- transition fine：在 1.75 全成功和 2.0 部分失败之间扫描 1.80、1.85、1.90、1.95（IDC=1）。

### 4.3 Region 判据（仅审计标签，不写入正式环境）

- C：OPF success rate < 95%，或成功解出现 configured voltage violation；
- B：不属于 C，且 minimum lower margin ≤0.02 pu、相对基准缩小 ≥50%，或 branch loading 达到基准 1.5×；
- A：其余稳定、宽松点。

上侧 generator setpoint margin 近零不单独触发 B，避免把正常调压点误判为 load stress。

## 5. Stress sweep results

下表 `loss` 是成功小时的平均 network loss；失败点不以 0 伪装物理结果。所有成功小时在当前 configured limits 下的 voltage/line/trafo violation 都为 0。

| background | IDC | OPF rate | min V | min lower margin | max line % | max trafo % | mean loss MW | max LMP spread | IDC incident % | Region |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.00 | 1.0 | 1.000 | 1.0146 | 0.0746 | 1.253 | 0.448 | 9.251 | 4.63 | 0.381 | A |
| 1.00 | 1.5 | 1.000 | 1.0146 | 0.0746 | 1.253 | 0.449 | 9.255 | 4.63 | 0.385 | A |
| 1.00 | 2.0 | 1.000 | 1.0147 | 0.0747 | 1.253 | 0.448 | 9.258 | 4.63 | 0.388 | A |
| 1.00 | 2.5 | 1.000 | 1.0148 | 0.0748 | 1.253 | 0.448 | 9.262 | 4.64 | 0.391 | A |
| 1.00 | 3.0 | 1.000 | 1.0148 | 0.0748 | 1.253 | 0.449 | 9.266 | 4.64 | 0.395 | A |
| 1.00 | 4.0 | 1.000 | 1.0149 | 0.0749 | 1.254 | 0.450 | 9.273 | 4.65 | 0.401 | A |
| 1.25 | 1.0 | 1.000 | 1.0157 | 0.0757 | 1.292 | 0.516 | 9.765 | 5.14 | 0.516 | A |
| 1.50 | 1.0 | 1.000 | 0.9781 | 0.0381 | 1.334 | 0.677 | 10.423 | 6.02 | 0.655 | B |
| 1.50 | 1.5 | 1.000 | 0.9778 | 0.0378 | 1.335 | 0.680 | 10.427 | 6.02 | 0.658 | B |
| 1.50 | 2.0 | 1.000 | 0.9776 | 0.0376 | 1.335 | 0.683 | 10.431 | 6.03 | 0.661 | B |
| 1.50 | 3.0 | 1.000 | 0.9771 | 0.0371 | 1.335 | 0.689 | 10.439 | 6.04 | 0.668 | B |
| 1.75 | 1.0 | 1.000 | 0.9442 | 0.0042 | 1.357 | 0.782 | 11.216 | 391.46 | 0.717 | B |
| 1.80 | 1.0 | 0.833 | 0.9444 | 0.0044 | 1.358 | 0.784 | 11.391 | 171.69 | 0.717 | C |
| 1.85 | 1.0 | 0.708 | 0.9444 | 0.0044 | 1.356 | 0.778 | 11.397 | 182.98 | 0.716 | C |
| 1.90 | 1.0 | 0.542 | 0.9448 | 0.0048 | 1.348 | 0.751 | 11.470 | 121.27 | 0.691 | C |
| 1.95 | 1.0 | 0.458 | 0.9443 | 0.0043 | 1.351 | 0.761 | 11.438 | 329.04 | 0.707 | C |
| 2.00 | 1.0 | 0.375 | 0.9442 | 0.0042 | 1.356 | 0.776 | 11.357 | 274.63 | 0.712 | C |
| 2.00 | 1.5 | 0.375 | 0.9442 | 0.0042 | 1.355 | 0.776 | 11.360 | 287.30 | 0.713 | C |
| 2.00 | 2.0 | 0.375 | 0.9442 | 0.0042 | 1.355 | 0.776 | 11.363 | 300.67 | 0.714 | C |
| 2.00 | 3.0 | 0.375 | 0.9441 | 0.0041 | 1.353 | 0.776 | 11.370 | 329.78 | 0.719 | C |
| 2.50 | 1.0 | 0.000 | — | — | — | — | — | — | — | C |
| 2.50 | 1.5 | 0.000 | — | — | — | — | — | — | — | C |
| 2.50 | 2.0 | 0.000 | — | — | — | — | — | — | — | C |
| 2.50 | 3.0 | 0.000 | — | — | — | — | — | — | — | C |
| 3.00 | 1.0 | 0.000 | — | — | — | — | — | — | — | C |

当前 configured thermal limit 下没有任何 branch violation。另行计算的 `baseline peak × 1.2/1.5` exceedance 只是相对压力指标：例如 background=1.5、IDC=1 时分别有 167/80 个 line-hour exceedance。它们不能称为 thermal violation，也没有写入 formal capacity。C 区的 exceedance 数因成功小时数下降而不可与 A/B 直接比较。

### 5.1 敏感性

Background 增加总体呈合理变化：loss 9.25→10.42→11.22 MW，transformer max 0.448→0.677→0.782%，最低电压在 1.5 后明显下降，1.75 接近 0.94 下边界，之后 OPF success 快速下降。line max 只从 1.253 增至约 1.357%，说明 thermal denominator 对压力不敏感。LMP spread 在 1.75 附近爆增，是接近可行域边界的经济/数值信号，但不适合作为硬安全定义。

IDC 增加可辨识但很弱：background=1 时，IDC incident max loading 0.381→0.401%，IDC bus minimum voltage 1.053107→约1.052994 pu，loss 9.251→9.273 MW，IDC LMP 上升。全网最低电压略升是 OPF redispatch 的系统级结果，不否定 IDC bus 局部压降。IDC×4 也只把不到 1.04 MW 的基准净负荷增至不到约 4.16 MW，相对 259 MW 原生负荷仍小。

诊断图：

- `outputs/grid_stress_audit_phase2_5/background_vs_min_voltage.png`
- `outputs/grid_stress_audit_phase2_5/background_vs_max_branch_loading.png`
- `outputs/grid_stress_audit_phase2_5/idc_vs_idc_bus_voltage.png`
- `outputs/grid_stress_audit_phase2_5/idc_vs_local_loading.png`
- `outputs/grid_stress_audit_phase2_5/coarse_2d_heatmaps.png`

完整机器可读结果：`grid_stress_audit.json`、`case_summaries.csv`、`hourly_results.csv`、3 类 capacity CSV 与 bus voltage observation CSV，均在上述输出目录。

## 6. Region classification

### Region A：Normal / Loose

已验证的主要范围是 background 1.0–1.25；background=1 时 IDC 1–4 全部属于 A。OPF 稳定，lower voltage margin 大，configured branch loading 很低。

### Region B：Stressed but Feasible

**稳健候选：background=1.5，IDC=1–3。** 四个二维点全部 24/24 成功，minimum lower margin 约 0.037–0.038 pu，局部 loading、loss、LMP 均有可识别变化。

**强压力上沿：background=1.75，IDC=1。** 仍为 24/24，但最低电压 0.9442 pu、lower margin 仅 0.0042 pu，LMP spread 极大。这个点适合作为边界审计，不宜直接作为默认长期训练中心。

### Region C：Infeasible / Unstable

background=1.80 开始出现 4/24 失败；1.85、1.90、1.95、2.0 的失败持续增加；2.5 与 3.0 全部失败。上限由电压/无功/发电可行性先触发，不是 9900 MVA thermal limit。

### Q5 与 Q6

- **Q5：background load 首先形成压力。** IDC×4 仍为 A；background 到 1.5 已进入 B，1.8 进入 C。
- **Q6：stressed-but-feasible 区间。** 当前数据支持 background≈1.5、IDC=1–3 作为最有用候选；background=1.75、IDC=1 是更靠近突破边界的上沿点。未扫描的 1.75×更高 IDC 组合不能凭空外推为可行。

## 7. Safe RL implications

### Q2：line / transformer capacity 能否直接作为 Safe RL constraint？

**No（作为真实 thermal safety constraint）；partially（只作为潮流响应观测或相对压力特征）。** OPF 确实 enforce 它们，但统一 9900 MVA 与异常 `max_i_ka/sn_mva` 不具物理校准依据。在完成重标定前，用 100% loading 定义 Safe RL 成功/失败会使约束几乎永不激活。

### Q3：bus voltage limits 是否可靠？

**相对 branch thermal limits 更可靠，且必须逐 bus。** 当前每个 bus 有显式边界，bus 5/7 的上限为合法 generator setpoint 扩展到 1.07/1.09；因此 trajectory 中 1.06–1.09 pu 并非自动 violation。未来约束应逐 bus 使用真实 `min_vm_pu/max_vm_pu`，同时把 generator setpoint 贴上限与负荷欠压 margin 区分，不能统一用 0.95–1.05。

### Q4：AC OPF 当前真正 enforce 什么？

逐 bus voltage、gen P/Q、ext-grid P/Q、15 条 line `RATE_A`、5 台 transformer `RATE_A` 均 enforce。差别在于 voltage/generator feasibility 在扫描中实际形成边界；branch thermal constraints 虽存在，但 9900 MVA 使其不 binding。

### Q7：最值得未来 Safe RL 约束的量（按当前证据排序）

1. **OPF failure / feasibility**：最直接，background 1.80 后立即出现且随压力增长；但应分解原因，避免只给 agent 一个黑箱 failure flag。
2. **逐 bus voltage lower/upper margin**：当前最清晰、连续且接近边界的物理信号，尤其 lower margin；必须 bus-specific。
3. **Generator/ext-grid P/Q headroom（建议新增诊断量）**：本阶段虽不在用户给定四项候选中，但它很可能解释电压边界和 OPF failure，物理意义高于当前 branch loading。
4. **Line loading**：保留观测价值；完成可信 thermal recalibration 后才升格为硬约束。
5. **Transformer loading**：同上；当前 rating 同样不可信，且扫描最高仍不足 0.8%。

## 8. Recommended next step

### Q8：是否需要重新标定 branch thermal limits？

**需要，但本阶段不执行。** 可选择以下路径，并在正式写入前单独论证：

1. **公开 benchmark rate 路径**：寻找明确带 branch ratings 的 IEEE-14/MATPOWER 变体，记录版本、字段语义、base MVA 与转换过程，优先使用可复现公开数据；不能只因为能产生 violation 就采用。
2. **工程参数路径**：按电压等级、导线/电缆类型、变压器铭牌、环境条件计算 continuous/emergency rating；同时校验 pandapower 低压侧等值表示，避免把 0.208 kV 数值机械解释为物理线路。
3. **基准潮流安全因子路径（仅实验 reference）**：`limit = baseline peak × 1.2 或 1.5`，用于做算法敏感性和相对拥塞实验。它必须明确标注 synthetic/reference，不能写成真实 thermal limit，也不能与 configured violation 混报。

建议下一阶段先增加 generator/ext-grid P/Q headroom 审计，并对 background 1.5–1.8、IDC 1–3 做更密的二维边界扫描；在 branch rating 获得外部依据前，不恢复 grid reward、Safe RL 或 Lagrangian。

## 9. Tests

执行：

```text
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe -m pytest marl\tests\test_grid_stress_sweep.py -q
```

结果：`5 passed`。pandapower 报告 64 条 `tap_dependency_table` deprecation warning，不影响结果。

| Test | 结果 | 证据 |
|---|---|---|
| 1. diagnostic 不修改 formal default config | PASS | 深拷贝比较 ENV/REWARD/DATA/GRID/GRID_SCENARIO，运行审计及 probe 后完全相等。 |
| 2. sweep 前后 git diff 不自动改正式参数 | PASS | 前后正式 tracked diff 均仅为用户既有 4 文件、36 insertions/4 deletions；没有新增自动改写。 |
| 3. 同 seed + 同 multiplier deterministic | PASS | 24h `PhysicsHour` 完全相等；重复 OPF summary/hour rows 完全相等。 |
| 4. 1/1 复现 Phase 2 量级 | PASS | 24/24；V=1.01457–1.08954；line=1.2531%；trafo=0.4480%。 |
| 5. 无 NaN/Inf | PASS | 内存递归检查；JSON 使用 `allow_nan=False`；失败数据用 JSON null / CSV 空字段。 |
| 6. failure 不使 sweep 崩溃 | PASS | background=100 的单小时测试返回 rate=0、failure_count=1 和失败 message；全 sweep 在 0/24 点继续完成。 |

## 10. Git

- HEAD before：`e777d841e21892b97971486129d1095b7d3d149c`
- 当前分支：`codex-model-improvement`
- 本阶段新增：`diagnostics/grid_stress_sweep.py`、`marl/tests/test_grid_stress_sweep.py`、本报告；生成结果位于 ignored `outputs/grid_stress_audit_phase2_5/`。
- 建议/实际 commit subject：`test: audit grid limits and stress operating region`
- Commit after SHA：见本任务最终回复（Git commit 无法在自身内容中自引用最终 SHA）。
- 用户既有 dirty changes 未纳入本阶段提交：`.gitignore`、`configs/config_ultimate.py`、`env_wrappers/grid_coupled_env.py`、`grid_model/grid_cache.py` 及既有 untracked 开发文件。

至此停止；未恢复 grid reward、Safe RL、Lagrangian，未修改正式 branch capacity。
