# MAPPO短训练前检查——第一部分最小修正实施报告

## 结论

第一部分最小修正已经完成。仓库现在具有不依赖 pytest、不导入 `marl/tests`、不使用运行时 monkeypatch 的正式双 Agent MAPPO 入口。正式入口使用项目环境工厂创建环境，并在同一个 HARL runner 生命周期内调用标准 `runner.run()` 连续执行配置指定的 update 数量。

本次只真实运行了 1-update 核心验收，没有启动 40-update MAPPO短训练，也没有进入缓存、完整日志、固定评估、分代 checkpoint、多 worker、HAPPO、HGTA 或 Safe RL 阶段。

## 1. 修改文件

### 新增

- `configs/harl_mappo_short.yaml`
  - 正式 MAPPO 基础配置。
  - 固定已验证的 smoke 参数。
  - 显式设置 `use_bounded_box_actions: true`、`action_aggregation: prod`、`share_param: false`。
  - 分开保存用户 `updates` 与运行时派生的 `num_env_steps`。
- `marl/envs/harl_env_factory.py`
  - 正式单环境、train VecEnv、eval VecEnv 工厂。
  - 复用 `train/train_ppo_ultimate.py::make_unmonitored_env()`。
  - 第一版明确只支持一个 rollout worker。
- `marl/runners/__init__.py`
  - 导出项目薄 runner 适配层。
- `marl/runners/idc_mappo_runner.py`
  - 薄适配 `OnPolicyMARunner`，只替换 stock runner 无法注入外部 VecEnv 的构造阶段。
  - rollout、buffer insert、return/GAE、MAPPO actor update、centralized critic update、save 均继承固定 HARL 实现。
  - 增加严格有限值与动作边界检查，不做 clip。
- `train/train_harl_mappo_short.py`
  - 正式 CLI、yaml 解析、CLI 覆盖、参数校验、环境校验、标准 runner 生命周期、最终保存、resolved config 与 metadata 快照。
- `marl/tests/test_harl_mappo_formal_entry.py`
  - 覆盖配置派生、正式环境工厂、禁止测试模块依赖和真实 1-update 正式流程。
- `docs/dev_reports/MAPPO_SHORT_TRAINING_PART1_MINIMAL_IMPLEMENTATION.md`
  - 本报告。

### 修改

- `.gitignore`
  - 仅新增 `!configs/harl_mappo_short.yaml`，使正式 yaml 可以纳入 Git。

### 保持不动

- `envs/idc_price_env.py`
- `env_wrappers/grid_coupled_env.py`
- `marl/envs/idc_grid_multi_agent_env.py`
- `marl/adapters/action_adapter.py`
- `marl/bridges/harl_bridge.py`
- `marl/bridges/harl_padded_bridge.py`
- `marl/diagnostics/bess_virtual_action_monitor.py`
- `marl/tests/test_harl_mappo_smoke.py`
- `C:\Users\bulio\Desktop\IDC\HARL` 中的 MAPPO、GAE、buffer、有界 Box 动作和保存实现

实施前已经存在的两个未跟踪报告文件也未改动：

- `docs/dev_reports/MAPPO_SHORT_TRAINING_PRECHECK_PART1_CALL_CHAIN_CONFIG.md`
- `docs/dev_reports/MAPPO_SHORT_TRAINING_READINESS_AUDIT.md`

## 2. 正式调用链

```text
python -m train.train_harl_mappo_short
→ train/train_harl_mappo_short.py::main()
→ configure_import_paths()
→ load_yaml_config()
→ resolve_config()
→ derive_num_env_steps()
→ marl/envs/harl_env_factory.py::make_harl_train_env()
→ _make_harl_vec_env()
→ make_harl_single_env()
→ train/train_ppo_ultimate.py::make_unmonitored_env()
→ IDCPriceEnv20D
→ GridCoupledEnv
→ IDCGridMultiAgentEnv
→ HarlIDCGridBridge
→ HarlPaddedBridge
→ HARL ShareDummyVecEnv
→ validate_environment_contract()
→ marl/runners/idc_mappo_runner.py::IDCOnPolicyMARunner(..., train_envs=...)
→ HARL OnPolicyBaseRunner.run()
→ warmup()
→ collect()
→ VecEnv.step()
→ insert()
→ compute() / GAE
→ OnPolicyMARunner.train()
→ IDC actor update
→ BESS actor update
→ centralized critic update
→ after_update()
→ train/train_harl_mappo_short.py::run_training() 最终 runner.save()
→ actor_agent0.pt
→ actor_agent1.pt
→ critic_agent.pt
```

`IDCOnPolicyMARunner` 没有复制或改写 MAPPO update、loss、GAE、buffer 或保存算法。之所以存在该适配层，是固定 HARL 的 stock runner 构造器只能调用其模块全局环境工厂，不能直接接收项目预构造 VecEnv；正式代码又明确禁止修改 HARL 全局 `make_train_env`/`get_num_agents`。

## 3. 配置说明

### 3.1 来源与覆盖顺序

```text
configs/harl_mappo_short.yaml
→ CLI显式覆盖
→ derive_num_env_steps()强制重新派生
→ validate_resolved_config()配置约束
→ validate_environment_contract()真实环境空间约束
→ resolved_config.json运行时快照
```

正式入口不加载 HARL 默认 `mappo.yaml`，不加载单 Agent `PPO_CONFIG`，也不调用 smoke 的 `_smoke_algo_args()`。

### 3.2 CLI

正式入口支持：

- `--config`
- `--seed`
- `--updates`
- `--episode-length`
- `--rollout-threads`
- `--output-dir`
- `--scenario`
- `--device`
- `--harl-source`
- `--runtime-path`

### 3.3 updates换算

```text
num_env_steps = updates × episode_length × n_rollout_threads
```

入口会校验所有因子为正整数，并验证派生步数可以按固定 HARL 的：

```text
num_env_steps // episode_length // n_rollout_threads
```

精确还原用户要求的 update 数量，不允许静默丢弃余数。

示例：

- `updates=5, episode_length=24, n_rollout_threads=1` → `num_env_steps=120`
- `updates=5, episode_length=24, n_rollout_threads=2` → `num_env_steps=240`

第一版环境工厂仍明确拒绝实际启动 `n_rollout_threads != 1`；第二个值只验证换算规则，为后续多 worker 阶段保留接口。

### 3.4 强制校验

正式训练开始前会真实构造并 reset 环境，校验：

- `n_agents == 2`
- Agent 顺序为 `idc, bess`
- observation 维度为 `288, 288`
- action 维度为 `22, 22`
- centralized state 维度为 `294`
- 两个动作空间 low 全为 `0`
- 两个动作空间 high 全为 `1`
- `use_bounded_box_actions is True`
- `state_type == EP`
- `episode_length == 24`
- `action_aggregation == prod`
- `share_param is False`

动作越界时立即报错，不做 clip。

### 3.5 快照与元数据

每次正式运行目录包含：

- HARL 自带 `config.json`
- 项目生成的 `resolved_config.json`
- 项目生成的 `run_metadata.json`
- `logs/`
- `models/actor_agent0.pt`
- `models/actor_agent1.pt`
- `models/critic_agent.pt`

`run_metadata.json` 分开记录：

- 项目实际 Git HEAD 或明确 Git 错误；
- HARL 实际 Git HEAD 或明确 Git 错误；
- 上游 HARL 基线 commit `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`；
- 当前预期 HARL HEAD `050ad6a294fe9f7572985dea910d59ea6d4f94b4`；
- Python、PyTorch、device；
- seed、updates、episode length、线程数和 num env steps；
- scenario、Agent 顺序和各空间维度；
- bounded Box 开关和 action aggregation；
- 开始/结束时间、状态、输出目录、模型路径；
- 完成 update 数、有限值检查和动作边界检查结果。

Git 命令不可用时记录 `head: null` 和明确错误，不伪造 commit；能读取 HARL HEAD 且与固定 HEAD 不符时拒绝启动。

## 4. 测试结果

### 4.1 新增正式入口完整测试

```powershell
$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:HARL_RUNTIME_PATH=(Join-Path (Get-Location).Path '.tmp_harl_runtime')
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -m pytest marl/tests/test_harl_mappo_formal_entry.py -q -s
```

最终结果：

- 退出码：`0`
- `5 passed`
- `0 failed`
- `209 warnings`
- 耗时：`43.45s`
- 真实执行：24 步 rollout、1 次 update、两个 actor 更新、centralized critic 更新、最终三个模型文件保存
- NaN/inf：未发现
- 动作越界：未发现

warnings 由 pandapower 旧网络数据的 `tap_dependency_table` deprecation warning 和 pytest cache 无写权限 warning 构成；最终正式 runner 已无 tensor 标量转换 warning。

### 4.2 最终元数据单用例复验

```powershell
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -m pytest 'marl/tests/test_harl_mappo_formal_entry.py::HarlMAPPOFormalEntryTest::test_one_update_formal_cli' -q -s
```

结果：

- 退出码：`0`
- `1 passed`
- `0 failed`
- `201 warnings`
- 耗时：`39.55s`
- `run_metadata.json` 包含 `state_dim=294`
- `run_metadata.json` 包含固定的 `harl_expected_git_head`
- 三个最终模型文件均存在

### 4.3 双Agent、Bridge、padding和动作适配回归

```powershell
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -m pytest marl/tests/test_action_adapter.py marl/tests/test_multi_agent_shapes.py marl/tests/test_multi_agent_parity.py marl/tests/test_multi_agent_rollout.py marl/tests/test_harl_bridge_shapes.py marl/tests/test_harl_bridge_parity.py marl/tests/test_harl_bridge_rollout.py marl/tests/test_harl_padding_shapes.py marl/tests/test_harl_padding_observation.py marl/tests/test_harl_padding_action_invariance.py marl/tests/test_harl_padding_parity.py marl/tests/test_bess_virtual_action_monitor.py -q
```

结果：

- 退出码：`0`
- `17 passed`
- `0 failed`
- `7 subtests passed`
- `873 warnings`
- 耗时：`137.21s`

### 4.4 固定HARL bounded Box动作测试

```powershell
$env:PYTHONPATH='C:\Users\bulio\Desktop\IDC\HARL;' + (Get-Location).Path
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -m pytest 'C:\Users\bulio\Desktop\IDC\HARL\tests\test_bounded_box_actions.py' -q
```

结果：

- 退出码：`0`
- `7 passed`
- `0 failed`
- `3 subtests passed`
- 耗时：`2.37s`

### 4.5 原MAPPO smoke回归

```powershell
$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:HARL_RUNTIME_PATH=(Join-Path (Get-Location).Path '.tmp_harl_runtime')
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -m pytest marl/tests/test_harl_mappo_smoke.py -q -s
```

结果：

- 退出码：`0`
- `1 passed`
- `0 failed`
- `202 warnings`
- 耗时：`51.38s`
- 原 smoke 的保存、重新加载和 BESS 诊断目标保持通过

### 4.6 说明

开发中的第一次新增测试在受限沙箱内导入 `.tmp_harl_runtime` 时只能看到无内容的 namespace `yaml`，结果为 `2 passed, 3 failed`，没有进入环境或训练。入口随后增加了 `yaml.safe_load`、`tensorboardX.SummaryWriter`、`setproctitle.setproctitle` 的真实 API 校验；在能够读取现有依赖文件的进程中完成以上最终验收。期间没有联网或安装依赖。

## 5. 首次正式1-update命令

以下命令只运行一个 update，不是 40-update 短训练：

```powershell
cd C:\Users\bulio\Desktop\IDC\ultimate_simplify

$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:HARL_RUNTIME_PATH=(Join-Path (Get-Location).Path '.tmp_harl_runtime')
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'

& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' `
  -m train.train_harl_mappo_short `
  --config configs/harl_mappo_short.yaml `
  --seed 7110 `
  --updates 1 `
  --episode-length 24 `
  --rollout-threads 1 `
  --output-dir runs/idc_bess_mappo_short `
  --scenario idc_bess_padding `
  --device cpu
```

该命令使用正式环境工厂与标准 `runner.run()`，不是 pytest，也不调用 smoke 函数。

## 6. 有意未完成事项

以下内容按要求留到后续阶段，本次没有实现或启动：

- OPF缓存检查；
- 随机种子体系全面检查；
- 完整训练、动作、BESS policy diagnostics 和物理指标日志；
- optimizer 状态保存与精确断点续训；
- update 10/20/30/40 分代 checkpoint；
- update 0/10/20/30/40 固定场景评估；
- final 模型重载后的正式确定性评估；
- 多 worker 并行环境；
- 3 seeds × 40 updates MAPPO短训练；
- HAPPO smoke与短训练；
- HGTA；
- Safe RL。

## 7. 当前阶段判断

“MAPPO短训练前检查——第一部分：训练调用链与配置来源”的阻塞项已经修正。仓库已具备正式单 seed、单 rollout worker、多 update 连续训练入口；开始真正的 40-update 短训练前，仍应按计划继续第二部分及后续日志、checkpoint、固定评估和运行保障检查。
