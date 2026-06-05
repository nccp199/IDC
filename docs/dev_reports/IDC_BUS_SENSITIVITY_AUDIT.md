# IDC 接入节点敏感性扫描报告

生成时间：2026-05-27  
范围：IEEE14 单 IDC 新增负荷接入节点扫描；不训练 PPO，不修改 reward，不改 GridCoupledEnv，不改 IEEE14 原始负荷。

## 0. 当前结论摘要

本阶段只新增了独立诊断脚本 `scripts/scan_idc_bus_sensitivity.py`，并运行了两轮扫描：

- 第一轮：AC OPF，cache off，MEF off，扫描所有原始负荷 bus，IDC 额外负荷为 `0/5/10/20/30/50/80/100/150/200 MW`；若 200 MW 仍成功，额外扫 `250/300 MW`。
- 第二轮：对 Top-5 敏感 bus 做 MEF 精扫，负荷点为 `0/20/50/100/150/200 MW`。

核心结论：

- 当前 bus9 不是最敏感节点，但也不是完全无反馈节点。bus9 可行到 200 MW，250 MW AC OPF 失败；若采用 `minV >= 0.97` 软裕度，建议 bus9 正常训练负荷先放在 30-100 MW，150 MW 作为轻 stress。
- 最敏感的纯负荷节点是 bus14、bus12，其次是 bus10、bus11、bus13。bus14 在 100 MW 失败，bus12/11/13 在 150 MW 失败，bus10 在 200 MW 失败。
- 最稳健但反馈较弱的节点是 bus2、bus4、bus5；bus2/4/5 到 300 MW 仍成功，LMP 变化也最小。bus2 是发电机+负荷节点，若要做“纯负荷接入”实验，不建议作为首选。
- 电网约束先紧的是电压/无功可行性，不是线路或变压器容量。所有成功点的 `maxLine` 最高约 1.99%，`maxTrafo` 最高约 1.49%，远低于 100%。线路/变压器容量在当前 IEEE14 参数下几乎不构成约束。
- LMP 对负荷最敏感的节点排序大致为：bus12、bus14、bus3、bus10、bus6、bus9。bus12 在 100 MW 处 LMP 跳到 11243，属于贴近 OPF 边界的异常强反馈，不适合当默认训练点。
- MEF 精扫显示：bus3 的 MEF 随负荷变化比较清晰；bus13/14 相对稳定；bus12 在边界附近不稳定；bus10 在 150 MW 出现极端负 MEF，建议视为近边界数值/调度切换信号，不作为正常训练反馈。
- 后续多 IDC 推荐先选 2-3 个纯负荷节点：bus9 + bus10 + bus13 或 bus9 + bus10 + bus11，每个 20-80 MW，总新增 IDC 负荷先控制在 80-180 MW。bus12/bus14 留作 stress 场景。

## 1. 新增诊断脚本

新增文件：

- `scripts/scan_idc_bus_sensitivity.py`

脚本特点：

- 直接调用 `grid_model.solve_opf()`，不经过 `GridCoupledEnv`。
- 不使用 grid cache；`--no-cache` 仅作为 CLI 显式标记。
- 不训练 PPO，不读写 PPO baseline，不修改 reward。
- 输出 bus 清单、逐负荷 OPF 结果、bus 汇总、Top-K MEF 精扫和 Markdown 摘要。

已运行命令：

```powershell
python -m scripts.scan_idc_bus_sensitivity --opf-mode ac --no-cache --no-mef
python -m scripts.scan_idc_bus_sensitivity --opf-mode ac --no-cache --top-k-mef 5 --mef-loads-mw 0,20,50,100,150,200
```

输出文件：

- `outputs/diagnostics/idc_bus_inventory.csv`
- `outputs/diagnostics/idc_bus_sensitivity_scan.csv`
- `outputs/diagnostics/idc_bus_sensitivity_summary.csv`
- `outputs/diagnostics/idc_bus_sensitivity_mef_topk.csv`
- `outputs/diagnostics/idc_bus_sensitivity_summary.md`

## 2. IEEE14 bus 分类

| IEEE bus | pandapower index | P MW | Q Mvar | 原始负荷 | 发电机 | slack | 当前 IDC 默认 |
|---:|---:|---:|---:|---|---|---|---|
| 1 | 0 | 0.0 | 0.0 | 否 | 否 | 是 | 否 |
| 2 | 1 | 21.7 | 12.7 | 是 | 是 | 否 | 否 |
| 3 | 2 | 94.2 | 19.0 | 是 | 是 | 否 | 否 |
| 4 | 3 | 47.8 | -3.9 | 是 | 否 | 否 | 否 |
| 5 | 4 | 7.6 | 1.6 | 是 | 否 | 否 | 否 |
| 6 | 5 | 11.2 | 7.5 | 是 | 是 | 否 | 否 |
| 7 | 6 | 0.0 | 0.0 | 否 | 否 | 否 | 否 |
| 8 | 7 | 0.0 | 0.0 | 否 | 是 | 否 | 否 |
| 9 | 8 | 29.5 | 16.6 | 是 | 否 | 否 | 是 |
| 10 | 9 | 9.0 | 5.8 | 是 | 否 | 否 | 否 |
| 11 | 10 | 3.5 | 1.8 | 是 | 否 | 否 | 否 |
| 12 | 11 | 6.1 | 1.6 | 是 | 否 | 否 | 否 |
| 13 | 12 | 13.5 | 5.8 | 是 | 否 | 否 | 否 |
| 14 | 13 | 14.9 | 5.0 | 是 | 否 | 否 | 否 |

本轮优先扫描原始负荷且非 slack 的 bus：

`[2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]`

bus7 是零负荷中间节点，bus8 是发电机节点，bus1 是 slack 节点，本轮不作为默认 IDC 候选。

## 3. AC OPF 逐节点承载能力

| bus | max OPF 成功 MW | first failed MW | minV>=0.97 最大 MW | 最小成功 minV | maxLine % | maxTrafo % | LMP range | 建议角色 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 12 | 100 | 150 | 50 | 0.9400 | 1.2447 | 0.5983 | 11203.33 | normal_or_moderate_stress |
| 14 | 80 | 100 | 50 | 0.9400 | 1.2450 | 0.7203 | 68.69 | stress_candidate |
| 3 | 250 | 300 | 200 | 0.9400 | 1.4777 | 0.7952 | 118.40 | normal_training_candidate |
| 10 | 150 | 200 | 100 | 0.9400 | 1.2537 | 1.0457 | 39.02 | normal_or_moderate_stress |
| 13 | 100 | 150 | 100 | 0.9836 | 1.2461 | 0.6465 | 6.13 | normal_or_moderate_stress |
| 11 | 100 | 150 | 100 | 0.9874 | 1.2478 | 0.6860 | 6.23 | normal_or_moderate_stress |
| 6 | 250 | 300 | 200 | 0.9532 | 1.2739 | 1.4467 | 38.59 | normal_training_candidate |
| 9 | 200 | 250 | 150 | 0.9400 | 1.2731 | 1.4869 | 17.42 | normal_training_candidate |
| 4 | 300 | 未失败 | 300 | 0.9863 | 1.5202 | 1.0004 | 6.32 | normal_training_candidate |
| 5 | 300 | 未失败 | 300 | 0.9958 | 1.2835 | 0.9758 | 5.35 | normal_training_candidate |
| 2 | 300 | 未失败 | 300 | 1.0156 | 1.9894 | 0.9566 | 5.22 | normal_training_candidate |

解释：

- `max OPF 成功 MW` 是硬约束下 AC OPF 成功的最大扫描点。
- `minV>=0.97 最大 MW` 是建议训练用的软裕度指标。因为当前 pandapower IEEE14 bus 下限可到 0.94，硬成功不代表有良好安全裕度。
- 多个节点在高负荷时 minV 正好贴到 0.94，这说明 OPF 先碰电压下界，再进入不可行或不收敛。

## 4. 哪些 bus 更敏感

按硬失败门槛看：

1. bus14：80 MW 成功，100 MW 失败，最敏感。
2. bus12、bus11、bus13：100 MW 成功，150 MW 失败。
3. bus10：150 MW 成功，200 MW 失败。
4. bus9：200 MW 成功，250 MW 失败。
5. bus3、bus6：250 MW 成功，300 MW 失败。
6. bus2、bus4、bus5：300 MW 仍成功，最不敏感。

按 `minV >= 0.97` 软安全裕度看：

- 只能到 50 MW：bus12、bus14。
- 可以到 100 MW：bus10、bus11、bus13。
- 可以到 150 MW：bus9。
- 可以到 200 MW：bus3、bus6。
- 可以到 300 MW：bus2、bus4、bus5。

工程含义：

- 如果目标是让 grid feedback 明显但不频繁失败，bus9、bus10、bus11、bus13 比较合适。
- 如果目标是 stress 测试，bus12 和 bus14 很合适。
- 如果目标是稳定训练但电网反馈弱，bus4、bus5 可以作为低敏感 anchor。
- bus2、bus3、bus6 同时是发电机 bus 和负荷 bus，技术上可扫，但作为“新增 IDC 纯负荷接入点”的解释不如 bus9/10/11/13/4/5 清晰。

## 5. 电压、线路、变压器谁先紧

结论：电压先紧，线路和变压器几乎不紧。

证据：

- bus14 在 80 MW 成功时 minV 已贴近 0.94，100 MW 失败。
- bus12 在 100 MW 成功时 minV 贴近 0.94，150 MW 失败，且 LMP 出现极端跳变。
- bus9 在 200 MW 成功时 minV 贴近 0.94，250 MW 失败。
- 所有成功点 `maxLine` 最高只有约 1.99%，`maxTrafo` 最高约 1.49%，远离 100%。

因此，当前 IEEE14 约束紧度诊断和上一阶段一致：OPF 成功率下降主要来自电压/无功可行域，而不是热稳定或线路容量。

## 6. LMP 敏感性

| rank | bus | LMP at 0 MW | LMP at max OK | LMP range | LMP slope/MW |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 40.37 | 11243.70 | 11203.33 | 74.9008 |
| 2 | 14 | 41.16 | 109.85 | 68.69 | 0.7463 |
| 3 | 3 | 40.58 | 158.98 | 118.40 | 0.2873 |
| 4 | 10 | 40.30 | 79.32 | 39.02 | 0.2001 |
| 5 | 6 | 39.74 | 78.33 | 38.59 | 0.0968 |
| 6 | 9 | 40.15 | 57.57 | 17.42 | 0.0645 |
| 7 | 11 | 40.15 | 46.38 | 6.23 | 0.0600 |
| 8 | 13 | 40.56 | 46.69 | 6.13 | 0.0593 |
| 9 | 4 | 40.19 | 46.51 | 6.32 | 0.0168 |
| 10 | 2 | 38.37 | 43.59 | 5.22 | 0.0166 |

注意：

- bus12 在 100 MW 附近出现 LMP 极端值，说明已经贴近 OPF 边界，不能把 11243 当作正常训练价格信号。
- bus14、bus10、bus9 的 LMP 变化更适合作为训练反馈，其中 bus14 更偏 stress，bus9/bus10 更适合主训练。

## 7. MEF Top-K 精扫

Top-K 自动选择：bus12、bus14、bus3、bus10、bus13。

MEF 汇总优先使用正负荷点，避免 0 MW 时 minus 扰动被 clamp 到 0 导致敏感性虚高。

| bus | MEF mean min | MEF mean max | MEF plus/minus 最大差 | 诊断 |
|---:|---:|---:|---:|---|
| 10 | -4167.36 | 175.20 | 26.93 | 150 MW 贴边时出现极端负 MEF，建议只信任 <=100 MW |
| 12 | 65.01 | 153.76 | 16.78 | 50-100 MW 附近边界明显，100 MW MEF 已失败 |
| 14 | 166.00 | 166.50 | 0.20 | 正负荷成功点 MEF 较稳定，但 100 MW OPF 失败 |
| 13 | 128.98 | 157.45 | 0.05 | MEF 稳定，适合作为温和反馈节点 |
| 3 | 138.41 | 236.33 | 0.04 | MEF 随负荷变化清晰，但 bus3 是发电机+负荷节点 |

建议：

- 正常训练不要使用 bus10 的 150 MW MEF、bus12 的 100 MW LMP/MEF 作为稳定反馈。
- 若做 MEF 驱动的 grid feedback，优先从 bus9、bus10 <=100 MW、bus13、bus3/6 可选节点开始。
- 需要专门做“边界/高碳敏感 stress”时，再启用 bus12、bus14。

## 8. 哪些 bus 适合正常训练，哪些适合 stress

### 正常训练优先

| 推荐等级 | bus | 建议单 IDC 负荷范围 | 理由 |
|---|---:|---|---|
| 高 | 9 | 30-100 MW，150 MW 轻 stress | 当前默认节点，纯负荷 bus，200 MW 前可行，反馈适中 |
| 高 | 10 | 20-80 MW，100 MW 轻 stress | LMP 比 bus9 更敏感，200 MW 失败，适合训练可见反馈 |
| 中高 | 13 | 20-80 MW，100 MW 上限 | 纯负荷 bus，MEF 稳定，150 MW 失败 |
| 中高 | 11 | 20-80 MW，100 MW 上限 | 纯负荷 bus，电压软裕度到 100 MW |
| 中 | 4 | 50-150 MW | 纯负荷 bus，很稳但 LMP/电压反馈偏弱 |
| 中 | 5 | 50-150 MW | 纯负荷 bus，很稳但反馈偏弱 |

### 可选但解释需谨慎

| bus | 建议单 IDC 负荷范围 | 备注 |
|---:|---|---|
| 3 | 50-150 MW，200 MW stress | 发电机+负荷 bus；MEF/LMP 变化明显 |
| 6 | 50-150 MW，200 MW stress | 发电机+负荷 bus；可行域较大 |
| 2 | 50-150 MW | 发电机+负荷 bus，300 MW 仍成功但反馈很弱 |

### Stress 场景

| bus | stress 范围 | 备注 |
|---:|---|---|
| 14 | 50-80 MW | 100 MW 失败，适合测试电压边界 |
| 12 | 50-100 MW | 100 MW 成功但 LMP 极端，150 MW 失败 |
| 10 | 100-150 MW | 150 MW 贴边，MEF 可能不稳定 |
| 9 | 150-200 MW | 200 MW 成功但 minV 已贴边，250 MW 失败 |

### 暂不建议默认接入

- bus1：slack bus，作为 IDC 默认接入点会混淆平衡功率和节点负荷效应。
- bus7：零负荷中间节点，本轮不是原始负荷 bus。
- bus8：发电机节点且无原始负荷，本轮不是主候选。

## 9. 多 IDC 初始推荐

建议先做 2-3 个 IDC，不要一上来接很多节点。

### 方案 A：主训练，纯负荷节点，反馈适中

- IDC-A：bus9，40-80 MW。
- IDC-B：bus10，20-50 MW。
- IDC-C：bus13 或 bus11，20-50 MW。

总新增 IDC 负荷约 80-180 MW。该方案不修改 IEEE14 原始负荷，且不默认使用 stress 节点。

### 方案 B：主训练 + 稳定 anchor

- IDC-A：bus9，40-80 MW。
- IDC-B：bus10，20-50 MW。
- IDC-C：bus4 或 bus5，50-100 MW。

bus4/5 反馈较弱但可行域宽，可以作为稳定负荷 anchor。

### 方案 C：stress 专用

- bus14，50-80 MW。
- bus12，50-100 MW。
- bus9，150-200 MW。
- bus10，100-150 MW。

该方案只用于 stress 诊断或 Safe RL 约束成本验证，不建议作为默认 PPO 训练场景。

## 10. 对本阶段问题的直接回答

1. 哪些 bus 适合接入 IDC？  
   主训练优先 bus9、bus10、bus13、bus11；稳定弱反馈可选 bus4、bus5；bus3/6 可选但属于发电机+负荷节点。

2. 哪些 bus 对 IDC 负荷更敏感？  
   bus14、bus12 最敏感；bus10、bus11、bus13 次之；bus9 中等；bus2/4/5 最不敏感。

3. 5/10/20/30/50/80/100/150/200 MW 是否成功？  
   详见 `outputs/diagnostics/idc_bus_sensitivity_scan.csv`。摘要：bus14 到 80 MW 成功、100 MW 失败；bus12/11/13 到 100 MW 成功、150 MW 失败；bus10 到 150 MW 成功、200 MW 失败；bus9 到 200 MW 成功、250 MW 失败。

4. 哪些 bus 优先导致电压接近边界？  
   bus12、bus14、bus10、bus9、bus3、bus6。多个高负荷成功点 minV 已贴到 0.94。

5. 哪些 bus 优先导致线路或变压器接近边界？  
   当前没有。maxLine 最高约 1.99%，maxTrafo 最高约 1.49%，远离热极限。

6. 哪些 bus 的 LMP 对负荷更敏感？  
   bus12、bus14、bus3、bus10、bus6、bus9。bus12 的极端 LMP 属于边界信号，不应作为正常价格反馈。

7. 哪些 bus 的 MEF 更敏感？  
   Top-K 中 bus3 的 MEF 负荷响应较清晰；bus12、bus10 在边界附近出现不稳定；bus13/14 在成功正负荷点较稳定。

8. 哪些 bus 适合作为正常训练场景？  
   bus9、bus10、bus13、bus11、bus4、bus5。推荐先用 20-100 MW 区间，不要直接贴硬边界。

9. 哪些 bus 只适合作为 stress 场景？  
   bus14、bus12；bus10 的 150 MW、bus9 的 150-200 MW 也应视为 stress 或边界诊断。

10. 后续多 IDC 推荐接几个、接哪些节点、每个节点多少 MW？  
    先接 2-3 个：bus9 40-80 MW，bus10 20-50 MW，bus13 或 bus11 20-50 MW。需要稳定 anchor 时加入 bus4/5 50-100 MW。需要 stress 时再加入 bus12/bus14。

## 11. 下一步建议

- 不建议马上改 reward 或训练 PPO；先确认多 IDC 接入方案和单 IDC MW 标定。
- 建议下一阶段做一个“多 IDC 负荷组合扫描”，只新增诊断脚本，扫描 bus9/bus10/bus13/bus11 的组合总负荷。
- SAFE RL 可以继续做，但它依赖足够紧的约束信号；建议先把正常训练负荷尺度从当前 0-3 MW 提升到至少几十 MW，否则 constraint cost 仍然长期为 0。
- HGTA/MAPPO 暂时不是最高优先级。当前瓶颈是负荷尺度和接入点选择，不是策略网络结构。
