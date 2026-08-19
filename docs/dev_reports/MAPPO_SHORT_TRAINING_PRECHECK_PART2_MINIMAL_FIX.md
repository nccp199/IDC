# MAPPO短训练前检查——第二部分最小修正实施报告

## 结论

本次限定的四项最小修正已经完成：启动校验改用独立 probe 环境；真实电网启动状态会在 runner 创建前校验；运行元数据明确区分 HARL 场景与物理实验场景；同一个实际 seed 被分别记录为 algorithm/environment/server/task seed。

正式1-update和原MAPPO smoke均真实运行并通过。本次没有开始MAPPO短训练，也没有改动物理模型、动作语义、reward、Bridge、padding、runner、HARL算法或配置超参数。

## 1. 修改文件

### 新增

- `docs/dev_reports/MAPPO_SHORT_TRAINING_PRECHECK_PART2_MINIMAL_FIX.md`：本实施报告。

### 修改

- `train/train_harl_mappo_short.py`
  - `validate_environment_contract()`现在明确只用于可丢弃 probe 环境。
  - 新增`_find_grid_env()`，只沿既有`.env`包装链读取`GridCoupledEnv`运行属性。
  - 新增`validate_grid_startup()`，从 probe reset 后的`last_info`和既有环境属性执行启动检查。
  - 新增`prepare_training_environment()`，负责 probe 构造、reset/check、关闭，以及 fresh train 环境重建。
  - `run_training()`改用上述准备流程；不再在正式训练环境上执行校验 reset。
  - 扩展`run_metadata.json`和`resolved_config.json`的运行时元数据。
- `marl/tests/test_harl_mappo_formal_entry.py`
  - 新增 probe 隔离、首任务序列一致性、对象独立性、正常 grid 启动、fallback 阻断、初始 OPF/MEF 失败阻断和元数据字段断言。

### 保持不动

- `marl/envs/harl_env_factory.py`
- `configs/harl_mappo_short.yaml`
- `envs/idc_price_env.py`
- `env_wrappers/grid_coupled_env.py`
- `marl/envs/idc_grid_multi_agent_env.py`
- `marl/adapters/action_adapter.py`
- `marl/bridges/harl_bridge.py`
- `marl/bridges/harl_padded_bridge.py`
- `marl/runners/idc_mappo_runner.py`
- `grid_model/**`
- `C:\Users\bulio\Desktop\IDC\HARL`中的MAPPO、GAE、buffer和有界Box实现

## 2. Probe流程

正式启动流程现在是：

```text
独立 probe VecEnv
→ validate_environment_contract(): reset并检查空间、维度、有限值
→ validate_grid_startup(): 检查真实grid/OPF/MEF启动状态
→ close probe VecEnv
→ 使用相同配置和seed重新构造fresh train VecEnv
→ 不reset train VecEnv
→ 创建IDCOnPolicyMARunner
→ runner.run()
→ HARL runner.warmup()执行train环境第一次reset
```

实现位于：

- `train/train_harl_mappo_short.py::prepare_training_environment()`
- `train/train_harl_mappo_short.py::run_training()`

probe关闭规则：

- 无论校验成功或失败，都尝试关闭 probe。
- 如果校验已经抛出原始异常，probe关闭异常不会覆盖该原始异常。
- 如果校验成功但probe关闭失败，关闭异常正常向上传播，不会继续创建runner。
- 不修改或重置任何内部RNG；probe关闭后直接重新构造全新训练环境。

### 首个task realization未被消耗的证据

测试使用同一seed `7110`比较：

```text
A：fresh baseline训练环境的第一次reset
B：独立probe完成reset/close后，新建fresh训练环境的第一次reset
```

A与B逐项一致：

- padded observations；
- centralized states；
- `lambda_t`完整序列；
- 全部Task的ID、类型、到达时间、时长、负载曲线、workload、deadline、priority和中断/并行属性；
- `total_task_count`；
- `initial_backlog_work`。

同时确认 fresh train 环境交给调用方时`last_info is None`，证明它尚未reset。

### 对象独立性证据

测试确认probe与train以下对象均不是同一实例：

- `ShareDummyVecEnv`；
- `HarlPaddedBridge`；
- `HarlIDCGridBridge`；
- `IDCGridMultiAgentEnv`；
- `GridCoupledEnv`；
- `IDCPriceEnv20D`；
- `GridResultCache`。

## 3. Grid启动检查

检查发生在probe reset完成后、fresh train环境和runner创建前。

| 检查项 | 实际来源 | 正常要求 | 失败行为 |
| --- | --- | --- | --- |
| grid scenario source | `HarlPaddedBridge.last_info["grid_scenario_source"]` | 不以`fallback`开头 | 抛出`RuntimeError` |
| CSV加载成功 | `grid_scenario_enabled`与实际source组合判断 | dynamic scenario启用且source非fallback | 抛出`RuntimeError` |
| grid启用 | `GridCoupledEnv.grid_enabled` | `True` | 抛出`RuntimeError` |
| OPF模式 | `GridCoupledEnv.opf_mode` | `ac`（大小写归一化后） | 抛出`RuntimeError` |
| MEF启用 | `GridCoupledEnv.use_mef` | `True` | 抛出`RuntimeError` |
| 初始OPF | `last_info["grid_opf_success"]` | `True` | 抛出`RuntimeError` |
| 初始MEF | `last_info["grid_mef_success"]` | `True` | 抛出`RuntimeError` |

失败消息包含：

- 失败检查项与实际值；
- `grid_scenario_source`；
- `grid_scenario_message`；
- `grid_opf_message`；
- `grid_mef_message`。

正式`main`配置实际通过，动态grid source为：

```text
C:\Users\bulio\Desktop\IDC\ultimate_simplify\data\grid_scenarios\nems_singapore\processed\nems_24h_load_scale.csv
```

对应message为：

```text
Loaded dynamic grid load scale.
```

测试没有修改真实配置文件。fallback通过测试替身在既有环境完成构造后设置运行状态，确认只创建probe一次即抛出异常，没有创建fresh train环境。初始OPF和MEF失败分别通过轻量运行状态替身验证。

## 4. Metadata新增字段

### `run_metadata.json`

| 字段 | 当前实际值 | 说明 |
| --- | ---: | --- |
| `harl_scenario` | `idc_bess_padding` | HARL协议/任务场景名 |
| `experiment_case` | `main` | 物理实验case |
| `grid_scenario_source` | NEMS 24小时CSV绝对路径 | 来自probe真实reset状态 |
| `grid_scenario_message` | `Loaded dynamic grid load scale.` | 来自probe真实reset状态 |
| `algorithm_seed` | `7110` | 当前HARL算法seed |
| `environment_seed` | `7110` | 当前传给正式环境工厂的seed |
| `server_seed` | `7110` | 工厂当前实际传给服务器模型的seed |
| `task_seed` | `7110` | 工厂当前实际传给任务模型的seed |

为保持兼容，原`scenario`和`seed`字段继续保留；其值分别等同于`harl_scenario`和`algorithm_seed`。

### `resolved_config.json`

上述八个字段同时写入`resolved_config.json`的`runtime`对象，表示在probe完成后确认的最终运行时事实。原始解析配置仍保留：

```text
env.scenario = idc_bess_padding
env.experiment_case = main
seed.seed = 7110
```

本次只拆分记录seed含义，没有改变seed行为、增加CLI seed参数或引入seed派生策略。

## 5. 测试结果

### 5.1 首次受限沙箱尝试

实际命令：

```powershell
$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -m pytest marl/tests/test_harl_mappo_formal_entry.py -q -s
```

结果：

| exit code | passed | failed | errors | warnings | pytest耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 9 | 0 | 1.35s |

该次未进入测试体。原因是受限沙箱无法读取既有`.tmp_harl_runtime`目录内容，Python只能看到空的namespace模块，`setUpClass`报告`yaml`、`tensorboardX`和`setproctitle`缺少API。没有安装依赖，也没有据此修改源码。

### 5.2 正式入口及本次新增测试

实际命令：

```powershell
$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:HARL_RUNTIME_PATH='C:\Users\bulio\Desktop\IDC\ultimate_simplify\.tmp_harl_runtime'
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -m pytest marl/tests/test_harl_mappo_formal_entry.py -q -s
```

结果：

| exit code | passed | failed | warnings | pytest耗时 | 工具总耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 9 | 0 | 257 | 54.62s | 59.9s |

覆盖结果：

- probe与train对象独立：通过；
- probe不消耗train首个task realization：通过；
- 正常main grid启动检查：通过；
- fallback在fresh train/runner之前阻断：通过；
- 初始OPF失败阻断：通过；
- 初始MEF失败阻断：通过；
- metadata字段和值：通过；
- 正式1-update：通过；
- 两个actor、centralized critic保存：通过；
- NaN/inf和动作边界既有检查：通过。

### 5.3 原MAPPO smoke回归

实际命令：

```powershell
$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -m pytest marl/tests/test_harl_mappo_smoke.py -q -s
```

结果：

| exit code | passed | failed | warnings | pytest耗时 | 工具总耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 202 | 38.77s | 44.5s |

warning均来自既有pandapower弃用提示、smoke中的tensor转标量提示和pytest cache权限提示；没有新增测试失败。

### 5.4 未运行的测试

本次没有重跑上一阶段已经通过的全套环境/Bridge/padding parity测试，因为本次没有修改这些模块；不能将其计入本次测试结果。

## 6. 验收对应关系

| 验收项 | 结果 |
| --- | --- |
| train环境在runner接管前未reset | 通过；fresh train的`last_info is None` |
| 启动校验由独立probe完成 | 通过 |
| probe关闭后重新构造fresh train | 通过 |
| 正常main配置可启动 | 通过 |
| grid scenario fallback阻断启动 | 通过 |
| 初始OPF/MEF失败阻断启动 | 通过 |
| metadata区分两类scenario | 通过 |
| metadata分别记录四类seed | 通过 |
| 正式1-update通过 | 通过 |
| 原MAPPO smoke通过 | 通过 |
| 物理结果、动作语义、reward和算法保持不动 | 通过；相关文件未修改 |

## 7. 明确保留到后续的事项

本次未实施：

- IDC电价专项检查；
- cache专项检查或cache策略修改；
- 完整OPF/MEF训练期统计与每步fail-fast；
- algorithm/environment/server/task seed独立配置；
- 多worker与worker seed；
- 固定场景评估；
- 分代checkpoint或完整日志系统；
- MAPPO 5-update、40-update或任何短训练；
- HAPPO；
- HGTA；
- Safe RL；
- 超参数调整。

本实施到第二部分最小修正为止，不自动进入第三部分。
