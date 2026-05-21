# Grid Node Sensitivity Report

## 1. Script

新增脚本：

- `scripts/scan_ieee14_bus_sensitivity.py`

运行方式：

```powershell
python -m scripts.scan_ieee14_bus_sensitivity
```

支持参数：

- `--idc-load-mw`
- `--delta-p-mw`
- `--out-dir`
- `--case-name`
- `--skip-ac`
- `--skip-dc`
- `--verbose`

## 2. Outputs

已生成：

- `data/grid_node_sensitivity/ieee14_node_sensitivity_static.csv`
- `data/grid_node_sensitivity/ieee14_node_sensitivity_detail.json`

CSV 为每个 IEEE14 bus 一行的静态 summary。JSON 保存 OPF、MEF、generator emission table、grid metrics 等更完整诊断信息，不包含 pandapower raw network。

## 3. Scan Parameters

- `case_name`: `ieee14`
- `idc_load_mw`: `1.0`
- `delta_p_mw`: `0.1`
- `out_dir`: `data/grid_node_sensitivity`
- DC OPF / DC-MEF: enabled
- AC OPF / AC-MEF: enabled

## 4. Result Overview

- Total scanned buses: 14
- DC OPF success: 14 / 14
- AC OPF success: 14 / 14
- DC-MEF success: 14 / 14
- AC-MEF success: 14 / 14
- OPF failed buses: None

## 5. Key Rankings

AC-MEF plus highest top 3:

1. IEEE bus 1, pandapower index 0: `284.036265 kgCO2/MWh`
2. IEEE bus 2, pandapower index 1: `235.247118 kgCO2/MWh`
3. IEEE bus 12, pandapower index 11: `225.329626 kgCO2/MWh`

AC-MEF plus lowest top 3:

1. IEEE bus 3, pandapower index 2: `154.653514 kgCO2/MWh`
2. IEEE bus 9, pandapower index 8: `168.421923 kgCO2/MWh`
3. IEEE bus 8, pandapower index 7: `173.230380 kgCO2/MWh`

AC LMP highest top 3:

1. IEEE bus 14, pandapower index 13: `41.238238`
2. IEEE bus 13, pandapower index 12: `40.623561`
3. IEEE bus 3, pandapower index 2: `40.591741`

AC max line loading highest top 3:

1. IEEE bus 2, pandapower index 1: `1.237626%`
2. IEEE bus 3, pandapower index 2: `1.235868%`
3. IEEE bus 4, pandapower index 3: `1.235649%`

## 6. Failures

No DC or AC OPF failures were observed in the 14-bus scan.

## 7. Notes

The scan computes MEF around the IDC-connected operating point. For each bus, the base OPF includes `idc_load_mw`, while plus/minus MEF perturbations use `idc_load_mw + delta_p_mw` and `idc_load_mw - delta_p_mw`.

The scan preserves the distinction between:

- IEEE bus number: original IEEE14 bus label, `1..14`
- pandapower bus index: internal pandapower index, `0..13`

## 8. Follow-Up Uses

- 固定单 IDC 主实验可选择一个代表性 bus，例如高 MEF、高 LMP、或中等水平 bus。
- 多 bus 扫描结果可用于节点敏感性分析。
- 后续多 IDC / MAPPO+CTDE 可用这些数据选择代表性接入节点。
- 单 IDC 节点扫描不能替代多 IDC 同时接入时的联合 OPF；多 IDC 场景仍需要联合扰动和联合 OPF。
