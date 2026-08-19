# MAPPO短训练前检查——第三部分MEF独立缓存规则实施报告

## 结论

MEF独立缓存规则已经按最小范围实施，原`0 MW`与`0.04 MW`互相复用导致的MEF minus观测错误已消除；OPF缓存规则、OPF/MEF算法、物理输入、reward和MAPPO管线保持不变。

第三部分重新判断：

> **2. 基本通过，仍有小范围近似风险。**

边界区域现在使用精确key，定向复测中MEF minus最大绝对误差由修正前的`126.2627604493 kg/MWh`下降到`0.0781229066 kg/MWh`。剩余误差来自`base_load > delta_p`区域有意保留的`0.01 MW`分箱，不是边界污染。所有正式回归通过，重复episode仍可把真实OPF调用从100次降至0次。

## 1. 修改文件

### 新增

- `marl/tests/test_grid_cache_formal_entry.py`
  - 固化OPF规则、MEF精确边界、独立分箱、delta、真实严重案例、命中、失败不缓存、正式环境隔离、24步等价和性能测试。
- `docs/dev_reports/MAPPO_SHORT_TRAINING_PRECHECK_PART3_MEF_CACHE_FIX.md`
  - 本实施报告。

### 修改

- `grid_model/grid_cache.py`
  - 新增`cache_mef_load_bin_mw`。
  - MEF key引入显式`exact`/`binned`负荷标签。
  - 边界区域使用6位精确负荷key。
  - 正常区域使用独立`0.01 MW`分箱。
  - `stats()`新增MEF负荷分箱字段。
  - 保留旧配置回退行为。
- `configs/config_ultimate.py`
  - 正式配置显式新增`"cache_mef_load_bin_mw": 0.01`。
- `env_wrappers/grid_coupled_env.py`
  - 显式旧配置缺少新键时回退到其OPF分箱。
  - step info新增`grid_cache_mef_load_bin_mw`。

### 保持不动

- `grid_model/mef_calculator.py`
- `grid_model/opf_solver.py`
- `envs/idc_price_env.py`
- Grid物理负荷与动态NEMS数据
- reward和grid reward
- Agent动作、Bridge、padding
- runner、MAPPO、GAE、有界Box动作
- HARL仓库
- IDC电价和实验超参数

## 2. 新MEF key规则

实现位置：

- `grid_model/grid_cache.py::make_mef_key()`
- `grid_model/grid_cache.py::_mef_load_key()`

规则为：

```text
base_load <= rounded(delta_p) + 1e-8
→ ("exact", round(base_load, cache_float_digits))

base_load > rounded(delta_p) + 1e-8
→ ("binned", quantize(base_load, cache_mef_load_bin_mw))
```

`1e-8`仅用于6位精度下的边界判断，不改变传给MEF或OPF的原始物理输入。

完整MEF key继续包含：

```text
(
    "mef",
    opf_mode,
    bus,
    hour,
    quantized_grid_load_scale,
    ("exact" | "binned", load_value),
    rounded_delta_p,
)
```

实际示例：

```text
0.04 MW, delta=0.1:
("mef", "ac", 8, 0, 0.91, ("exact", 0.04), 0.1)

0.201 MW, delta=0.1:
("mef", "ac", 8, 2, 1.0, ("binned", 0.2), 0.1)

0.204 MW, delta=0.1:
("mef", "ac", 8, 2, 1.0, ("binned", 0.2), 0.1)

0.206 MW, delta=0.1:
("mef", "ac", 8, 2, 1.0, ("binned", 0.21), 0.1)
```

`delta_p`仍位于key末尾；相同`base_load=0.08`时：

```text
delta=0.10 → ("exact", 0.08)
delta=0.05 → ("binned", 0.08)
```

两个key还分别包含`0.10`和`0.05`的delta，因此不会互相命中。

### OPF key保持不变

OPF仍使用原结构：

```text
("opf", mode, bus, hour, load_scale_bin, opf_load_bin)
```

实测：

```text
OPF key(0.00 MW) == OPF key(0.04 MW)
== ("opf", "ac", 8, 0, 1.0, 0.0)
```

没有修改OPF key、OPF算法或LMP提取。

## 3. 配置与统计字段

| 配置/字段 | 最终值或语义 | 位置 |
| --- | ---: | --- |
| `cache_load_bin_mw` | `0.1 MW`，仅用于OPF净负荷key | `configs/config_ultimate.py` |
| `cache_mef_load_bin_mw` | `0.01 MW`，仅用于MEF正常区域 | `configs/config_ultimate.py` |
| `cache_load_scale_bin` | `0.005`，OPF/MEF共同使用 | 现有配置 |
| `cache_float_digits` | `6` | `GridResultCache`内部默认 |
| `delta_p_mw` | `0.1 MW`，未修改 | `GRID_CONFIG` |
| 边界epsilon | `1e-8` | 只参与key区域判断 |
| `stats()["cache_load_bin_mw"]` | `0.1`，继续表示OPF bin | 保留 |
| `stats()["cache_mef_load_bin_mw"]` | `0.01` | 新增 |
| `grid_cache_load_bin_mw` | step info中的OPF bin | 保留 |
| `grid_cache_mef_load_bin_mw` | step info中的MEF bin | 新增 |
| `grid_cache_load_scale_bin` | step info中的load-scale bin | 保留 |

### 向后兼容

- `GridResultCache()`完全不传配置时采用新默认`0.01 MW`。
- 正式`GRID_CACHE_CONFIG`显式采用`0.01 MW`。
- 显式传入旧配置但缺少`cache_mef_load_bin_mw`时，MEF bin回退到该旧配置的`cache_load_bin_mw`。
- 实测旧配置`cache_load_bin_mw=0.2`时，运行时`cache_mef_load_bin_mw=0.2`，并写回可检查的resolved cache config，不是不可追踪的硬编码。

## 4. 原问题复测

测试状态：AC-OPF、bus index 8、hour 0、实际动态load scale `0.9100530292`、delta `0.1 MW`。

| base load | MEF负荷key | 首次查询 | MEF minus (kg/MWh) |
| ---: | --- | --- | ---: |
| 0.00 MW | `("exact", 0.0)` | miss | 0.0000000000 |
| 0.03 MW | `("exact", 0.03)` | miss | 95.5742195618 |
| 0.04 MW | `("exact", 0.04)` | miss | 127.4577660864 |
| 0.10 MW | `("exact", 0.1)` | miss | 319.0601987133 |

四个key互不相同，四次查询均独立计算并写入各自cache条目。

双向污染测试：

```text
顺序A：0.00 → 0.04 MW
命中：miss → miss
结果：0.00得到0；0.04得到自己的真实MEF minus

顺序B：0.04 → 0.00 MW
命中：miss → miss
结果：0.04得到自己的真实MEF minus；0.00仍得到0
```

在测试使用的`load_scale=1.0`下，`0.04 MW`真实MEF minus为`67.3716120090 kg/MWh`；正反顺序结果一致。修正前的“0先写入导致0.04直接返回0”不再发生。

完全相同的边界状态仍正常命中：

```text
0.04 MW第一次：miss
0.04 MW第二次：hit
MEF底层计算调用：1次
```

正常区域也继续复用：

```text
0.201 MW：miss
0.204 MW：hit（同0.20 MW的MEF bin）
MEF底层计算调用：1次
```

## 5. 测试结果

所有命令均设置：

```powershell
$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'
```

### 5.1 最终版新缓存测试

实际命令：

```powershell
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' `
  -m pytest marl/tests/test_grid_cache_formal_entry.py -q
```

| exit code | passed | failed | warnings | pytest耗时 | 工具总耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11 | 0 | 56 | 68.86s | 71.9s |

覆盖：

- 新默认值与旧配置回退；
- OPF key完全保持；
- `0/0.03/0.04/0.1 MW`精确MEF key；
- 正常区域`0.01 MW`分箱；
- delta进入key并改变边界区域；
- 真实AC-MEF双向严重案例；
- 相同状态与正常bin命中；
- OPF/MEF失败不缓存；
- probe/train/两个正式环境隔离；
- 24步cache on/off物理等价；
- 重复episode性能和solver计数。

在最后兼容分支调整前，同一完整测试也曾真实运行：`11 passed, 56 warnings, 72.10s`。最终版结果以上表为准。

### 5.2 兼容分支定向测试

实际命令：

```powershell
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' `
  -m pytest marl/tests/test_grid_cache_formal_entry.py::GridCacheFormalEntryTest::test_config_stats_and_legacy_fallback -q
```

| exit code | passed | failed | warnings | 耗时 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 0 | 0.31s |

### 5.3 正式MAPPO 1-update回归

实际命令：

```powershell
$env:HARL_RUNTIME_PATH='C:\Users\bulio\Desktop\IDC\ultimate_simplify\.tmp_harl_runtime'
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' `
  -m pytest marl/tests/test_harl_mappo_formal_entry.py -q -s
```

| exit code | passed | failed | warnings | pytest耗时 | 工具总耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 9 | 0 | 257 | 48.90s | 55.3s |

真实完成24步和1次update，保存两个actor及centralized critic；未出现NaN/inf，动作边界检查通过。

### 5.4 原MAPPO smoke回归

实际命令：

```powershell
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' `
  -m pytest marl/tests/test_harl_mappo_smoke.py -q -s
```

| exit code | passed | failed | warnings | pytest耗时 | 工具总耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 202 | 38.32s | 43.9s |

warning来自既有pandapower弃用提示、smoke tensor转标量提示和pytest cache权限提示，不是缓存修正失败。

### 5.5 定向误差脚本

使用临时PowerShell内联Python执行，不写入仓库：

- 退出码：0；
- 样本数：9；
- 工具总耗时：26.5s；
- OPF/MEF成功翻转：0；
- 电压/线路越限翻转：0。

## 6. 性能与数值结果

### 6.1 24步物理等价

cache关闭与修正后cache开启的第一个episode使用相同seed、相同task RNG初始状态和相同24步合法动作。

以下字段24步最大绝对差异全部为`0`：

- reward；
- SOC、BESS充放电功率；
- IDC/grid功率；
- cost、carbon；
- LMP、MEF plus、MEF minus；
- min/max voltage；
- max line loading；
- network loss；
- completion、backlog。

OPF success、MEF success和done也完全相同。

### 6.2 性能

| 模式 | 总耗时（reset+24步） | 主OPF调用 | MEF调用 | MEF内部OPF | 真实OPF总数 | 本episode hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache关闭 | 25.4850s | 25 | 25 | 75 | 100 | 0 |
| 修正后cache首次episode | 26.0102s | 25 | 25 | 75 | 100 | 0 |
| 修正后cache重复episode | 0.0218s | 0 | 0 | 0 | 0 | OPF 25、MEF 25 |

重复episode仍保留明确收益；独立MEF规则没有意外关闭缓存。

### 6.3 修正后分箱误差复测

9个样本覆盖：零负荷、`0–0.1 MW`多个值、`0.1 MW`边界、中/高负荷、最低grid scale、BESS充电、BESS放电近零负荷和PV高发。

| 指标 | 最大绝对误差 | 平均绝对误差 |
| --- | ---: | ---: |
| LMP | 0.0016943889 | 0.0004801695 |
| MEF plus (kg/MWh) | 0.0824281489 | 0.0095378093 |
| MEF minus (kg/MWh) | **0.0781229066** | **0.0090564835** |
| min voltage (p.u.) | 0.0000050130 | 0.0000013946 |
| max voltage (p.u.) | 0.0000146905 | 0.0000044273 |
| max line loading (%) | 0.0000359470 | 0.0000097992 |
| network loss (MW) | 0.0006434108 | 0.0001800403 |

说明：

- 边界区域MEF key不同，MEF plus/minus误差均为0。
- 表中剩余MEF误差只来自正常区域`0.01 MW`bin复用，测试偏移为`0.003 MW`。
- OPF仍按要求保留`0.1 MW`分箱，因此LMP、电压、线路和损耗误差与MEF边界精确规则无关。
- 没有OPF成功、MEF成功、电压越限或线路越限判断翻转。
- 当前grid reward关闭，grid reward误差为0。

## 7. 第三部分重新判断

选择：

> **2. 基本通过，仍有小范围近似风险。**

理由：

- 原严重边界错误已经消除，`0`和`0.04 MW`不再互相命中。
- 正常区域MEF独立分箱从`0.1 MW`缩小到`0.01 MW`。
- 修正后MEF minus最大抽样误差约`0.0781 kg/MWh`，相较修正前`126.26 kg/MWh`下降超过三个数量级。
- 24步首次运行保持完全物理等价，重复状态仍有显著性能收益。
- 仍然采用“同bin第一个原始输入结果”近似，所以不能宣称全状态空间完全无误差；本次9样本也不是大规模统计。

该结果不再支持“因明显MEF观测错误而必须关闭缓存”的判断。是否开始MAPPO短训练仍需等待后续检查或用户确认。

## 8. 未完成事项

本次明确未处理：

- IDC电价；
- 完整训练日志与OPF/MEF训练期统计；
- 多worker及worker seed；
- 固定场景评估；
- MAPPO短训练；
- HAPPO；
- HGTA；
- Safe RL；
- 后续第四部分检查。

未提交Git，完成后等待下一步确认。
