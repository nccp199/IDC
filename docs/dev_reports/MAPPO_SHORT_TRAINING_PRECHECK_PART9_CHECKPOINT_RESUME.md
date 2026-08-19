# MAPPO短训练前检查——第九部分：Checkpoint保存、加载与恢复训练语义

检查日期：2026-07-28  
项目：`C:\Users\bulio\Desktop\IDC\ultimate_simplify`  
HARL：`C:\Users\bulio\Desktop\IDC\HARL`  
验收环境：CPU、`torch_threads=1`、`n_rollout_threads=1`、`episode_length=24`、seed `7110`

## A. 现有保存能力审计

原保存链为：

```text
train/train_harl_mappo_short.py
→ IDCOnPolicyMARunner.run()
→ runner.save()
→ HARL OnPolicyBaseRunner.save()
→ models/actor_agent0.pt
→ models/actor_agent1.pt
→ models/critic_agent.pt
```

HARL原实现位于 `C:\Users\bulio\Desktop\IDC\HARL\harl\runners\on_policy_base_runner.py:724`，只调用三个网络的 `state_dict()`；加载位于同文件 `restore():742`，`load_state_dict()` 默认严格加载。当前 `use_valuenorm=false`，因此没有 ValueNorm 文件。

| 状态 | 原实现是否保存 | 原文件/键 | 原实现是否足以恢复训练 |
| --- | --- | --- | --- |
| IDC Actor参数 | 是 | `models/actor_agent0.pt` | 否，仅权重 |
| BESS Actor参数 | 是 | `models/actor_agent1.pt` | 否，仅权重 |
| Critic参数 | 是 | `models/critic_agent.pt` | 否，仅权重 |
| IDC optimizer | 否 | 无 | 否 |
| BESS optimizer | 否 | 无 | 否 |
| Critic optimizer | 否 | 无 | 否 |
| LR进度 | 否 | 无；仅当前optimizer含实际LR | 否 |
| ValueNorm/PopArt | 条件保存 | 当前关闭 | 当前无需；未来启用时需保存 |
| Update计数 | 否 | 无 | 否 |
| Global step | 否 | 无 | 否 |
| Episode计数 | 否 | 无 | 否 |
| Python/NumPy/PyTorch RNG | 否 | 无 | 否 |
| task/server Generator | 否 | 无 | 否 |
| 环境与下一episode状态 | 否 | 无 | 否 |
| Grid OPF/MEF cache | 否 | 无 | 否 |
| Logger计数 | 否 | 无 | 否 |
| BESS有效动作Mask语义 | 否 | 无 | 否 |
| 配置、数据、Grid指纹 | 否 | 无 | 否 |

结论：原 `models/` 是可靠的 model-only 权重产品，可用于推理、评估或明确标注的 warm start；它不是完整训练Checkpoint。

## B. 修改文件

新增：

- `marl/checkpointing/__init__.py`：Checkpoint公共API。
- `marl/checkpointing/environment_state.py`：显式、版本化环境状态序列化。
- `marl/checkpointing/training_checkpoint.py`：完整训练Checkpoint、原子保存、manifest、hash、兼容性校验和恢复。
- `marl/tests/test_training_checkpoint_resume.py`：12类、15个artifact验收测试。
- 本报告。

修改：

- `marl/runners/idc_mappo_runner.py`：总update语义、恢复后跳过warmup、post-update保存边界。
- `marl/logging/training_metrics.py`：logger状态、连续编号、分段与累计summary。
- `train/train_harl_mappo_short.py`：`--resume-checkpoint`、`--load-model-dir`、兼容性指纹、metadata和Checkpoint编排。
- `configs/harl_mappo_short.yaml`：正式Checkpoint配置段。

保持不动：

- `envs/idc_price_env.py`及物理step/reset语义。
- `env_wrappers/grid_coupled_env.py`及OPF/MEF调用逻辑。
- `grid_model/grid_cache.py`及cache key/命中规则。
- `marl/bridges/`、padding、BESS动作映射。
- MAPPO loss、GAE、Reward和BESS有效动作Mask算法。
- 固定HARL源码。

## C. Checkpoint Schema

Schema版本：`idc-mappo-training-resume-v1`，实现入口为 `marl/checkpointing/training_checkpoint.py:22` 和 `TrainingCheckpointManager:188`。

```text
checkpoint_schema_version
checkpoint_type = training_resume
algorithm = mappo
resume_boundary = post_update_only
mid_rollout_resume_supported = false
saved_at_update / global_step / episodes_completed
model_state
optimizer_state
normalizer_state
rng_state
environment_state
runner_state
logger_state
verification_state
compatibility
provenance
```

文件结构：

```text
checkpoints/
├── update_000001.pt
├── update_000001.manifest.json
├── latest.json
└── final.pt
```

`latest.json`只保存文件名、manifest、update和SHA-256指针，不使用Windows不兼容的符号链接。`final.pt`是最后update文件的原子复制，状态和hash与同run的最后update一致。

## D. 模型、优化器与Mask状态

- `model_state`按固定agent顺序保存 `actor_agent0_state_dict`、`actor_agent1_state_dict`、`critic_state_dict`；恢复始终 `strict=True`。
- `optimizer_state`保存两个Actor Adam与Critic Adam的完整 `state_dict`，包括param groups、LR、Adam step、`exp_avg`、`exp_avg_sq`、eps和weight decay。
- 当前不使用独立scheduler，`use_linear_lr_decay=false`；实际LR由optimizer状态恢复。为避免“checkpoint时未知未来总目标”造成线性衰减轨迹变化，当前正式精确恢复明确拒绝 `checkpoint.enabled=true` 与 `use_linear_lr_decay=true` 的组合。
- `normalizer_state.use_valuenorm=false`、`state_dict=null`；schema已为未来ValueNorm保留严格状态槽。
- Mask兼容契约逐项保存并严格比较：

```text
agent_order = [idc, bess]
padded_action_dims = [22, 22]
effective_action_dims = [22, 1]
virtual_action_dims = [0, 21]
effective_action_masks = [[1 × 22], [1, 0 × 21]]
action_padding_strategy = padding-v1-effective-mask
```

验收证明BESS模型仍为22维输出，有效维度仍为1，虚拟21维仍不参与优化。

## E. RNG与环境状态

完整Checkpoint保存：

- Python `random.getstate()`；
- NumPy全局RNG；
- PyTorch CPU RNG；
- CUDA RNG列表（仅CUDA可用时）；
- `IDCEnergyTaskModel.task_rng.bit_generator.state`；
- `IDCEnergyTaskModel.server_rng.bit_generator.state`；
- 已生成的服务器功率、算力和效率参数；
- `IDCPriceEnv20D`动态字段、完整Task静态/动态字段、BESS SOC/能量、平滑动作/负载、累计成本/碳/任务/峰值/throughput等；
- Grid场景曲线与来源；
- OPF、MEF两个LRU cache的顺序、结果、hits/misses和bin配置；
- `IDCGridMultiAgentEnv`、HARL Bridge、Padded Bridge的最近观测、动作、info、episode/global-step计数；
- 下一次collect所需actor/critic buffer index 0的obs、share_obs、RNN state、mask、active/bad mask和available actions。

实现位于 `marl/checkpointing/environment_state.py:124` 与 `load_environment_state_dict():168`。它序列化明确的数据字段，不pickle整个环境对象。

审计确认Gym `Space.np_random`和基类Gym `np_random`未被策略采样、task生成、server生成或物理transition消费，因此最终schema不保存这组构造期随机对象；真正影响训练的全局RNG、task/server Generator均保存并通过round-trip。

## F. 保存时机与原子性

真实顺序：

```text
24-step rollout
→ compute returns / GAE
→ IDC actor update
→ BESS actor update
→ centralized critic update
→ TrainingMetricsLogger.record_update()并flush update CSV
→ HARL episode_log
→ actor/critic buffer after_update()
→ TrainingCheckpointManager.maybe_save(update)
```

Runner实现位于 `marl/runners/idc_mappo_runner.py:184-244`。保存发生在 `after_update()`之后，因此：

- optimizer已经更新；
- step/episode/update日志已经写完；
- 没有未消费rollout；
- buffer index 0和VecEnv都指向下一episode的初始collect状态；
- Checkpoint与日志不会错位一个update。

原子保存实现位于 `training_checkpoint.py:272`：先写 `.tmp`、flush、`fsync`、`os.replace`，随后写manifest，最后更新`latest.json`。加载前检查文件存在、size和SHA-256；检查实现位于 `training_checkpoint.py:456`。验收目录无残留`.tmp`文件。

## G. Resume调用链

```text
train/train_harl_mappo_short.py --resume-checkpoint ... --updates 2
→ 构造新的train VecEnv与新runner（不reset正式train env）
→ 创建新的run/checkpoints目录
→ 校验manifest/size/SHA-256/type/schema
→ 严格比较compatibility
→ strict加载2 Actor + Critic
→ 加载3个optimizer与normalizer状态
→ 恢复显式环境/cache状态
→ 恢复buffer index 0
→ 恢复logger/update/global-step/episode计数
→ 最后恢复全局RNG
→ runner.resumed=true
→ runner.run()跳过warmup()/env.reset()
→ 从saved_update + 1继续
```

`--updates`在fresh和resume中始终表示“本run希望达到的总update数”。Checkpoint为update 1时，`--updates 2`只运行update 2；若目标不大于已保存update，启动即失败。

`--load-model-dir`只走原HARL model-only严格加载，明确标记为`model_only`，不恢复optimizer/RNG/env/logger；它与`--resume-checkpoint`互斥。model-only文件传给resume会因缺少training manifest而明确失败。

## H. 日志连续性

Logger状态接口位于 `marl/logging/training_metrics.py:228` 和 `load_state_dict():249`。

恢复写入新run目录，不append或覆盖父run。验收恢复段第一行：

```text
update = 2
global_step = 25
episode_id = 1
```

恢复段update行：

```text
update = 2
global_step = 48
episodes_completed = 2
```

恢复段`run_summary.json`：

```text
resumed = true
updates_completed_this_segment = 1
total_updates_completed = 2
environment_steps_this_segment = 24
episodes_completed_this_segment = 1
resume_parent = seed-07110-2026-07-28-20-42-30
```

新run metadata记录source、parent run、parent update、starting global step/episode、schema、边界和最终Checkpoint。

## I. 兼容性与错误处理

恢复前严格比较：algorithm、agent顺序、obs/state/action维度、Mask、episode length、rollout threads、share_param、action aggregation、有界Box开关、Actor/Critic网络、optimizer类型、Grid case、experiment case、Reward/Data/完整配置指纹、Grid CSV hash、cache bins、project/HARL HEAD。

训练seed由Checkpoint provenance显式保存；对本次早期验收artifact，loader兼容读取其`seed-07110-*` run id并仍严格核对当前seed。允许变化的字段只有输出目录、device映射、日志开关、总目标updates和Checkpoint间隔等非轨迹配置。

以下情况fail-fast：

- model-only权重冒充resume；
- manifest缺失；
- 文件size或SHA-256错误；
- schema/type/algorithm错误；
- Agent顺序、维度、Mask、网络、优化器、数据、Reward、Grid或cache配置不兼容；
- seed不同；
- `--updates <= saved_update`；
- optimizer、RNG、环境或logger状态缺失。

未实现危险的`--allow-incompatible-resume`静默绕过。

## J. 测试结果

### 1. 非训练配置回归

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" -m pytest `
  marl/tests/test_harl_mappo_formal_entry.py::HarlMAPPOFormalEntryTest::test_formal_sources_do_not_depend_on_test_modules `
  marl/tests/test_harl_mappo_formal_entry.py::HarlMAPPOFormalEntryTest::test_resolved_config_keeps_bounded_box_explicit `
  marl/tests/test_harl_mappo_formal_entry.py::HarlMAPPOFormalEntryTest::test_update_to_num_env_steps_derivation -q
```

结果：`3 passed, 1 warning in 2.29s`。

### 2. Part-9完整artifact测试

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  -m pytest marl/tests/test_training_checkpoint_resume.py -q
```

结果：`15 passed, 1 warning in 3.87s`。warning仅为pytest无法创建`.pytest_cache`，不影响源码、run或Checkpoint。

覆盖：模型严格加载与固定输入输出、Mask、三个optimizer、Python/NumPy/Torch RNG、环境/下一episode状态、update-2动作与Reward、精确延续、CSV连续编号、错误类型、损坏文件、不兼容配置、旧model-only权重。

### 3. Logger回归

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  -m pytest marl/tests/test_training_metrics_logger.py -q
```

结果：`13 passed, 17 warnings in 6.19s`。16条为pandapower既有DeprecationWarning，1条为pytest cache权限warning。

## K. 2-update精确恢复验收

### Run B父段：update 1

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" train/train_harl_mappo_short.py `
  --updates 1 --checkpoint-interval 1 `
  --output-dir runs/part9_resume_parent --device cpu `
  --harl-source "C:\Users\bulio\Desktop\IDC\HARL" `
  --runtime-path ".tmp_harl_runtime"
```

退出码0，耗时38.7秒。

### Run B恢复段：总目标update 2

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" train/train_harl_mappo_short.py `
  --updates 2 --checkpoint-interval 1 `
  --output-dir runs/part9_resumed_to_2 --device cpu `
  --resume-checkpoint "...\part9_resume_parent\...\checkpoints\update_000001.pt" `
  --harl-source "C:\Users\bulio\Desktop\IDC\HARL" `
  --runtime-path ".tmp_harl_runtime"
```

退出码0，只执行update 2，耗时36.0秒。

### Run A：不间断2 updates

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" train/train_harl_mappo_short.py `
  --updates 2 --checkpoint-interval 1 `
  --output-dir runs/part9_continuous_2 --device cpu `
  --harl-source "C:\Users\bulio\Desktop\IDC\HARL" `
  --runtime-path ".tmp_harl_runtime"
```

退出码0，耗时63.3秒。

Run目录：

- Run A：`C:\Users\bulio\Desktop\IDC\ultimate_simplify\runs\part9_continuous_2\gym\idc_bess_padding\mappo\idc_bess_mappo_short\seed-07110-2026-07-28-20-44-34`
- Run B父段：`C:\Users\bulio\Desktop\IDC\ultimate_simplify\runs\part9_resume_parent\gym\idc_bess_padding\mappo\idc_bess_mappo_short\seed-07110-2026-07-28-20-42-30`
- Run B恢复段：`C:\Users\bulio\Desktop\IDC\ultimate_simplify\runs\part9_resumed_to_2\gym\idc_bess_padding\mappo\idc_bess_mappo_short\seed-07110-2026-07-28-20-43-43`

| 对比项 | 最大差异 |
| --- | ---: |
| IDC Actor参数 | 0.0 |
| BESS Actor参数 | 0.0 |
| Critic参数 | 0.0 |
| IDC optimizer | 0.0 |
| BESS optimizer | 0.0 |
| Critic optimizer | 0.0 |
| Update 2动作 | 0.0 |
| Update 2 Reward | 0.0 |
| Reward分量 | 0.0 |
| Task指标 | 0.0 |
| BESS指标 | 0.0 |
| OPF/MEF/Cache | 0.0 |
| Actor/Critic训练指标 | 0.0（忽略耗时、路径、run id） |
| Mask诊断 | 0.0 |
| 下一episode/runner可消费状态 | 0.0 |

Checkpoint hash：

- 父update 1：`8fca5d489d80182c3d955ccf060b867183ce49824a01c8cac45851a45e327d7a`
- Run A update 2：`6066974c408912ca80e6dccc91881acb7858623973fec45957e0a20b0c0a913e`
- Run B update 2：`77c8f9352b3be8041bb541a92d43fc185578734f41e73e738fb4a3fae24bbeb7`

两个update-2文件因run provenance、logger run id和非轨迹metadata不同而整体hash不同；上述所有训练状态和结构化指标逐项相同。

## L. 文件大小与性能

| 项目 | 结果 |
| --- | ---: |
| Run A update-2 Checkpoint | 628,293 bytes |
| Run B update-2 Checkpoint | 628,421 bytes |
| 父update-1 Checkpoint | 582,085 bytes |
| 显式环境状态内存序列化估算 | 120,685 bytes |
| 环境/cache状态占Run A Checkpoint | 19.21% |
| Run A update-2保存耗时 | 0.0482 s |
| Run B恢复加载耗时 | 0.0338 s |
| update-2 OPF cache条目 | 49 |
| update-2 MEF cache条目 | 49 |
| OPF hits/misses | 2 / 49 |
| MEF hits/misses | 2 / 49 |

当前文件远小于1 MB，40 updates规模不需要异步或压缩保存。精确恢复必须保留cache；不保存cache只能提供“功能继续”，不能提供本次已证明的逐步完全复现。

说明：第一轮验收artifact曾包含未被消费的Gym space/base RNG快照；审计后最终schema按“只保存真正影响恢复的状态”移除了它们。task/server/global RNG、模型、optimizer、环境、cache、rollout和所有比较结果不受影响。

## M. 问题分类

### 阻塞MAPPO短训练

无。模型加载、完整恢复、Mask、环境下一状态、optimizer和损坏文件拒绝均已通过。

### 必须在40-update前修正

本部分发现的问题均已修正：此前只有权重、无optimizer/RNG/env/logger、恢复会额外reset、无hash、无兼容性检查、编号会重启的问题均不存在。

### 建议改进

- 任意环境step中途恢复；
- 异步/压缩/远端对象存储；
- 多worker状态序列化；
- 跨设备CUDA精确轨迹声明；
- 将Checkpoint schema迁移工具作为未来独立功能。

以上均不阻塞当前单worker CPU短训练。

### 可以保持现状

- 保留现有三个model-only权重文件；
- `use_valuenorm=false`；
- post-update-only恢复；
- Windows `latest.json`指针；
- 默认每5 updates保存、保留最近3个、保存final；
- 不以训练Reward定义best model。

## N. 第九部分判断

选择：**1. 通过，模型加载与update边界精确恢复均可靠。**

分项判断：

- 模型权重加载：通过。三个旧model-only文件严格兼容，固定输入输出一致。
- 完整训练恢复：通过。模型、optimizer、RNG、环境、cache、buffer、runner和logger均恢复。
- 精确等价性：通过。不间断2 updates与1+resume到2的所有训练相关比较项最大差异为0.0。
- 后续3 seeds × 40 updates支持：支持。可分别以三个seed启动独立run；中断后使用对应run的完整Checkpoint与相同seed恢复，`--updates 40`仍表示总目标。

## O. 最小后续建议

第十部分前无需继续修改Checkpoint。首次40-update短训练可直接使用：

```powershell
cd C:\Users\bulio\Desktop\IDC\ultimate_simplify
$env:HARL_SOURCE_PATH="C:\Users\bulio\Desktop\IDC\HARL"
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'

& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  train/train_harl_mappo_short.py `
  --seed 7110 `
  --updates 40 `
  --output-dir runs/mappo_short `
  --harl-source "C:\Users\bulio\Desktop\IDC\HARL" `
  --runtime-path ".tmp_harl_runtime"
```

若从update 20恢复到总目标40：

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  train/train_harl_mappo_short.py `
  --seed 7110 `
  --updates 40 `
  --resume-checkpoint "<parent-run>\checkpoints\update_000020.pt" `
  --output-dir runs/mappo_short_resumed `
  --harl-source "C:\Users\bulio\Desktop\IDC\HARL" `
  --runtime-path ".tmp_harl_runtime"
```

本报告止于第九部分，未进入第十部分、固定评估、HAPPO、Safe RL或正式40-update训练。
