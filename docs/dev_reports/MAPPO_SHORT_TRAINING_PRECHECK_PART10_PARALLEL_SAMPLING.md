# MAPPO短训练前检查——第十部分：并行采样正确性、性能与多Worker恢复

检查日期：2026-07-28  
项目：`C:\Users\bulio\Desktop\IDC\ultimate_simplify`  
HARL：`C:\Users\bulio\Desktop\IDC\HARL`  
正式入口：`train/train_harl_mappo_short.py`

## 结论摘要

第十部分通过。1、2、4 Worker 的正确性均通过；2 Worker 的相同 seed 重复运行和 post-update checkpoint 精确恢复均通过；Windows `spawn` 子进程能够正常创建、运行、异常回传和关闭。正式短训练推荐 **2 Worker**，专用配置 `configs/harl_mappo_short.yaml` 已更新为 `n_rollout_threads: 2`。

必须注意：`episode_length=24` 时，2 Worker 每次 update 使用 48 个 transitions。40 updates 对应 `num_env_steps=1920`，是 1 Worker、40 updates 的两倍样本量；后续 MAPPO/HAPPO 公平比较必须固定 Worker 数及总样本语义。

本部分未修改物理模型、Reward、MAPPO loss、GAE、BESS有效动作Mask、Grid/Safe目标、网络结构、学习率或求解精度，未运行40-update训练，也未进入第十一部分。

## A. 现有并行能力审计

| 层级 | 文件/函数 | 第十部分前行为 | 当前行为 | 多Worker兼容 |
| --- | --- | --- | --- | --- |
| CLI | `train/train_harl_mappo_short.py::build_parser()` | 已有 `--rollout-threads` | 保持CLI，允许1/2/4 | 是 |
| 配置解析 | `resolve_config()`、`derive_num_env_steps()` | 入口校验只允许1 | 校验允许1/2/4，并严格推导总步数 | 是 |
| 启动线程控制 | `configure_parallel_runtime()` | 无完整子进程BLAS限制 | 在数值库/子进程启动前设置4类BLAS变量为1 | 是 |
| 启动Probe | `prepare_training_environment()` | Probe与正式环境可能混用构造语义 | 独立1-Worker Probe关闭后，再创建未reset的正式VecEnv | 是 |
| 环境工厂 | `marl/envs/harl_env_factory.py::_make_harl_vec_env()` | 仅1 Worker `ShareDummyVecEnv` | 1返回Dummy；2/4返回项目子进程VecEnv | 是 |
| Worker工厂 | `make_harl_single_env()` | 单环境构造 | 顶层函数加`functools.partial`，可由Windows spawn序列化 | 是 |
| VecEnv | `marl/envs/parallel_vec_env.py::ProjectShareSubprocVecEnv` | 不存在 | 真正的每环境一个子进程；start method=`spawn` | 是 |
| Runner | `marl/runners/idc_mappo_runner.py::IDCOnPolicyMARunner` | 部分代码通过`envs[0]`检查顺序 | 从VecEnv公开元数据检查Agent顺序 | 是 |
| rollout | `IDCOnPolicyMARunner.run()/collect()` | HARL批次第一维已存在 | actions以Worker为第一维发送，仍执行finite/Box边界检查且不clip | 是 |
| buffer | HARL `OnPolicyBaseRunner.insert()` | 标准HARL | batch第一维保留Worker；critic仍使用`rewards[:, 0]` | 是 |
| 日志 | `marl/logging/training_metrics.py` | 已有per-worker accumulator | 经2/4 Worker真实产物验收 | 是 |
| checkpoint | `marl/checkpointing/` | 仅单环境状态 | 保存/恢复所有Worker状态并严格校验并行拓扑 | 是 |

### 关键问题回答

1. 原先限制为1 Worker的直接原因是正式入口和环境工厂的显式启动保护，而不是HARL buffer不支持batch。
2. 当前 `n_rollout_threads>1` 使用真实Windows子进程，不是串行DummyVecEnv。
3. 项目没有直接使用HARL上游的 `ShareSubprocVecEnv`；使用项目最小兼容层 `ProjectShareSubprocVecEnv`，原因是精确恢复需要项目环境专用的 `get_state/set_state` RPC、明确的child traceback和可验证的强制清理。
4. 环境thunk是顶层 `make_harl_single_env` 的 `functools.partial`，使用标准spawn pickle即可，不依赖闭包或pytest monkeypatch，也不要求cloudpickle。
5. `configure_import_paths()`加入的项目、固定HARL和runtime搜索路径由spawn继承。2/4 Worker真实运行也确认子进程能导入HARL并读取Grid CSV。
6. logger、TensorBoard、模型和checkpoint只存在于主进程Runner；child仅持有环境并响应RPC。
7. `train/train_harl_mappo_short.py` 的模块入口包含 `multiprocessing.freeze_support()`，未出现递归创建子进程。

## B. 修改文件

### 新增

- `marl/envs/parallel_vec_env.py`：Windows spawn VecEnv、worker本地自动reset、异常回传、状态RPC、资源清理。
- `marl/tests/test_parallel_rollout_sampling.py`：并行正确性、日志、异常、复现、checkpoint验收。
- `marl/tests/parallel_rollout_benchmark.py`：真实环境、真实OPF/MEF/cache、固定动作、固定96 transitions的无policy-update吞吐基准。

### 修改

- `marl/envs/harl_env_factory.py`：1/2/4 Worker选择、确定性seed公式、可pickle工厂及VecEnv元数据。
- `marl/runners/idc_mappo_runner.py`：从VecEnv元数据验证Agent顺序；保留HARL标准rollout/update。
- `marl/checkpointing/environment_state.py`：单Worker状态格式扩展为多Worker v2，保留单Worker v1兼容读取。
- `marl/checkpointing/training_checkpoint.py`：并行拓扑兼容清单、所有Worker状态保存/恢复、旧单Workercheckpoint迁移读取。
- `marl/checkpointing/__init__.py`：导出单环境状态序列化接口。
- `train/train_harl_mappo_short.py`：并行配置校验、Probe隔离、BLAS限制、元数据、freeze support和多Worker兼容清单。
- `configs/harl_mappo_short.yaml`：新增`parallel`段；基准完成后把专用短训练默认值更新为2 Worker。
- `marl/tests/test_harl_mappo_formal_entry.py`：默认配置步数期望改为2 Worker语义；显式单Worker隔离回归仍固定为1。

### 保持不动

- `IDCPriceEnv20D`、`GridCoupledEnv`、`IDCGridMultiAgentEnv`的物理与Reward逻辑。
- HARL上游Actor/Critic、MAPPO loss、GAE、buffer和网络结构。
- 有界Box动作分布及其log-prob数学。
- BESS有效动作Mask数学定义。
- OPF/MEF求解精度及cache key/共享规则（仍为每环境实例独立cache）。

## C. VecEnv与进程调用链

```text
train/train_harl_mappo_short.py::main()
→ configure_import_paths()
→ load_yaml_config() / resolve_config()
→ configure_parallel_runtime()
→ run_training()
→ prepare_training_environment()
   → make_harl_train_env(n=1)                 [独立、可丢弃Probe]
   → reset + Grid/OPF/MEF启动校验
   → close Probe
   → make_harl_train_env(n=正式Worker数)
→ marl/envs/harl_env_factory.py::_make_harl_vec_env()
   → n=1: HARL ShareDummyVecEnv
   → n=2/4: ProjectShareSubprocVecEnv
      → multiprocessing.get_context("spawn")
      → 每个rank执行_worker(partial(make_harl_single_env, worker_seed))
      → HarlPaddedBridge
      → HarlIDCGridBridge
      → IDCGridMultiAgentEnv
      → GridCoupledEnv
      → IDCPriceEnv20D
→ IDCOnPolicyMARunner.run()                    [主进程]
→ warmup()/collect()
→ ProjectShareSubprocVecEnv.step(actions)      [batch发往各child]
→ logger.per_step()
→ HARL OnPolicyBaseRunner.insert()
→ compute()                                    [returns/GAE保持HARL]
→ IDC actor update
→ BESS actor update（有效动作Mask保持生效）
→ centralized critic update
→ after_update()
→ TrainingCheckpointManager.maybe_save()
```

1 Worker没有child；2 Worker创建2个child；4 Worker创建4个child。主进程最终检查时 `python.exe` 进程数为0，未发现残留Worker。

## D. Worker seed与环境隔离

固定公式：

```text
worker_seed(rank) = base_seed + rank * worker_seed_stride
base_seed = 7110
worker_seed_stride = 1000
```

| Worker | environment seed | task seed | server seed |
| ---: | ---: | ---: | ---: |
| 0 | 7110 | 7110 | 7110 |
| 1 | 8110 | 8110 | 8110 |
| 2 | 9110 | 9110 | 9110 |
| 3 | 10110 | 10110 | 10110 |

结论：

- Worker 0可追溯到base seed；seed不依赖PID、时间或进程启动顺序。
- worker seed列表写入resolved config runtime、run metadata和checkpoint兼容清单。
- 每个child独立构造完整物理链、task RNG、server RNG、BESS状态和Grid cache。
- 同一2-Worker run内，Worker 0/1的任务和服务器参数确实不同；相同base seed与相同Worker数的两次运行又能精确复现。
- Probe是单独对象，在正式Worker创建前已关闭；Probe reset、任务生成和OPF/MEF访问不消耗正式Worker 0的RNG、cache或下一episode状态。
- 训练seed 7110系列与当前单Worker eval seed 2026不冲突。

## E. Shape、Reward和Mask验收

2 Worker和4 Worker均执行了真实环境reset和step：

| 数据 | 2 Worker | 4 Worker |
| --- | --- | --- |
| observation | `(2, 2, 288)` | `(4, 2, 288)` |
| centralized state | `(2, 2, 294)` | `(4, 2, 294)` |
| action | `(2, 2, 22)` | `(4, 2, 22)` |
| reward | `(2, 2, 1)` | `(4, 2, 1)` |
| done | `(2, 2)` | `(4, 2)` |
| info | Worker × Agent | Worker × Agent |

- 所有Worker均保持 `agent 0=IDC`、`agent 1=BESS`。
- 每个Worker内部 `IDC reward == BESS reward`，不同Worker允许不同。
- centralized critic沿用HARL `OnPolicyBaseRunner.insert()`的 `rewards[:, 0]`，没有把两Agent reward相加，也没有跨Worker求和或复制Worker 0。
- BESS在所有Worker仍为effective=1、padded=22、virtual=21。
- 多Worker update验收：`effective_physical_ratio_max_diff=0`、virtual mean gradient=0、virtual log-std gradient=0、virtual entropy optimization contribution=0。
- bounded Box动作在collect/存buffer/evaluate-actions链保持一致；Runner仍是越界即报错，未新增clip。

## F. 日志多Worker验收

### 2 Worker，1 update

- `step_metrics.csv`：48行。
- `episode_metrics.csv`：2行。
- `update_metrics.csv`：1行。
- `worker_id={0,1}`，每个Worker各有episode_step 0至23。
- `(worker_id, episode_id, episode_step)`唯一。
- 每个Worker的episode reward等于自身24个step reward之和。
- update rollout reward等于两个完整episode reward的算术平均值。
- 两个terminal分别归属各自Worker；global step=48，episodes completed=2。

### 4 Worker，1 update

- `step_metrics.csv`：96行。
- `episode_metrics.csv`：4行。
- `update_metrics.csv`：1行。
- `worker_id={0,1,2,3}`；global step=96，episodes completed=4。

TensorBoard global step使用全局transition计数。日志、TensorBoard与checkpoint均只由主进程写入，没有Worker同名覆盖或双计Reward。

## G. Cache隔离与冷/热表现

每个Worker checkpoint状态分别包含自己的OPF/MEF entries、hit/miss计数。未使用Manager共享cache、模块级可变cache或跨Worker写共享对象。

真实随机策略2-update结果如下。这里“update 2”仅表示已有cache可复用，策略动作变化意味着它不是完全相同输入的纯热cache。

| Worker数 | update | OPF hits/misses | MEF hits/misses | OPF hit rate | MEF hit rate | OPF/MEF成功率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 / 25 | 1 / 25 | 3.85% | 3.85% | 100% / 100% |
| 1 | 2 | 1 / 24 | 1 / 24 | 4.00% | 4.00% | 100% / 100% |
| 2 | 1 | 3 / 49 | 3 / 49 | 5.77% | 5.77% | 100% / 100% |
| 2 | 2 | 4 / 46 | 4 / 46 | 8.00% | 8.00% | 100% / 100% |
| 4 | 1 | 4 / 100 | 4 / 100 | 3.85% | 3.85% | 100% / 100% |
| 4 | 2 | 9 / 91 | 6 / 94 | 9.00% | 6.00% | 100% / 100% |

固定动作96-transition测试进一步确认：1 Worker cache为76 hits/25 misses；2 Worker每个cache各26/25；4 Worker每个cache各1/25。一个Worker的命中不会修改另一个Worker的计数。

## H. 异常传播与进程清理

- 构造、reset、step三个阶段分别注入child异常，主进程均收到包含Worker rank、命令、异常类型、消息和完整child traceback的 `RemoteWorkerError`。
- 未把异常吞掉后返回空Reward。
- 构造失败立即强制关闭已启动进程；运行中失败由正式入口finally关闭全部Worker。
- 轻量正式失败路径验收确认：`run_metadata.json.status=failed`、`run_summary.json.status=failed`、termination reason保留错误、未生成`final.pt`、所有child均已停止。
- 正常和异常测试后系统只读检查结果：`PYTHON_PROCESS_COUNT=0`。
- mock异步终止验证：某Worker提前done时只自动reset该Worker，`original_obs`没有写入其他Worker info；其他Worker继续自己的episode。

## I. 多Worker可复现性

条件：2 Worker、base seed=7110、1 update、相同配置，两次独立正式进程运行。

| 对比项 | 最大训练相关差异 |
| --- | ---: |
| Worker任务/服务器状态 | 0 |
| step actions | 0 |
| step Reward及分量 | 0 |
| episode Reward | 0 |
| 两个Actor参数/optimizer | 0 |
| Critic参数/optimizer | 0 |
| 结构化训练指标 | 0 |
| BESS Mask诊断 | 0 |

路径、run id、时间、PID等非训练字段被明确排除。update-1 checkpoint中的`total_updates_target`分别为2和1，它是运行计划字段，不是学习状态；忽略这一已知计划字段后update-1状态完全相同。update-2连续/恢复比较为严格训练状态0差异。

不要求1 Worker轨迹等于2 Worker轨迹，也不要求2 Worker轨迹等于4 Worker轨迹：batch shape、样本数和PyTorch RNG消费顺序均不同。

## J. 多Worker Checkpoint精确恢复

推荐方案为2 Worker，因此执行了强制的2 Worker精确恢复验收：

```text
Run A: 2 Worker连续训练到update 2
Run B: 2 Worker训练到update 1并保存
       → 新Python进程加载update_000001.pt
       → 不额外reset任何Worker
       → 继续到update 2
```

结果：

- 两个Actor、Critic和三个optimizer：最大差异0。
- update-2所有Worker actions、Reward、Reward分量、任务/BESS指标：最大差异0。
- 两个Worker的task/server RNG、BESS SOC、OPF/MEF cache与计数、Bridge/Padding状态、下一次collect观测：最大差异0。
- update编号、global step、episode编号、训练指标和Mask诊断：最大差异0。
- 未在恢复时reset或用seed重建状态冒充恢复。
- checkpoint严格记录并校验 `n_rollout_threads`、worker seed列表、VecEnv类型和start method；1↔2、2↔4不兼容恢复会被拒绝。
- 4 Worker不是最终推荐值，因此按任务约束未追加4 Worker精确恢复训练。

## K. 1/2/4 Worker性能表

真实双Agent MAPPO，CPU，base seed=7110，`torch_threads=1`，4类BLAS线程均为1，每个候选最多2 updates；物理、Reward、日志、checkpoint、OPF/MEF/cache配置相同。

| 指标 | 1 Worker | 2 Workers | 4 Workers |
| --- | ---: | ---: | ---: |
| transitions/update | 24 | 48 | 96 |
| 完整episodes/update | 1 | 2 | 4 |
| update 1 rollout time | 39.899 s | 26.649 s | 33.371 s |
| update 2 rollout time | 40.136 s | 24.926 s | 49.085 s |
| update 1 policy update time | 0.0245 s | 0.0140 s | 0.0199 s |
| update 2 policy update time | 0.0227 s | 0.0108 s | 0.0240 s |
| 2-update总wall time | 81.963 s | 56.565 s | 88.329 s |
| update 1 transitions/s | 0.601 | 1.800 | 2.875 |
| update 2 transitions/s | 0.598 | 1.925 | 1.955 |
| update 1 episodes/s | 0.0251 | 0.0750 | 0.1199 |
| update 2 episodes/s | 0.0249 | 0.0802 | 0.0815 |
| OPF solve count（两update misses） | 49 | 95 | 191 |
| MEF solve count（两update misses） | 49 | 95 | 194 |
| OPF总cache hit rate | 3.92% | 6.86% | 6.37% |
| MEF总cache hit rate | 3.92% | 6.86% | 4.90% |
| mean CPU usage | 未可靠采集 | 未可靠采集 | 未可靠采集 |
| peak memory | 未可靠采集 | 未可靠采集 | 未可靠采集 |
| child process count | 0 | 2 | 4 |
| warnings/errors | 无功能错误；pandapower弃用警告 | 同左 | 同左 |

CPU均值和峰值内存没有可靠采样器，因此不填造数据。4 Worker在update 2退化明显，说明OPF/MEF并行与CPU调度已出现竞争；其吞吐没有持续显著高于2 Worker。

## L. 固定样本预算比较

真实环境、真实OPF/MEF/cache、固定0.5动作、总计96 transitions、无policy update。该测试只衡量采样吞吐，不比较学习效果。

| 指标 | 1 Worker | 2 Workers | 4 Workers |
| --- | ---: | ---: | ---: |
| vector steps | 96 | 48 | 24 |
| creation | 2.789 s | 3.656 s | 4.739 s |
| reset | 1.078 s | 1.022 s | 1.383 s |
| rollout | 23.396 s | 25.419 s | 32.671 s |
| creation+reset+rollout | 27.263 s | 30.097 s | 38.793 s |
| rollout transitions/s | 4.103 | 3.777 | 2.938 |
| total transitions/s | 3.521 | 3.190 | 2.475 |
| rollout episodes/s | 0.1710 | 0.1574 | 0.1224 |
| OPF hits/misses | 76 / 25 | 52 / 50 | 4 / 100 |
| MEF hits/misses | 76 / 25 | 52 / 50 | 4 / 100 |
| alive after close | 无child | `[false,false]` | 4个均为`false` |

固定动作下1 Worker能在同一cache复用4个episode，因此固定总样本预算最快；这不能覆盖随机策略正式训练中2 Worker显著更高的实际rollout吞吐结论。两个结果必须同时保留。

## M. 推荐Worker数及样本量说明

推荐 **2 Worker**：

1. 2/4 Worker正确性都通过，但2 Worker吞吐从单Worker约0.60提升到1.80～1.92 transitions/s。
2. 4 Worker第二次update退化到约1.95 transitions/s，总wall time反而最长，固定96-transition基准也最慢。
3. 2 Worker日志、Mask、异常清理、同seed精确复现和checkpoint精确恢复均通过。
4. 2 Worker没有进程泄漏，OPF/MEF成功率均为100%。
5. 2 Worker相对4 Worker更适合后续MAPPO/HAPPO使用同一并行拓扑做公平对比。

样本语义：

```text
transitions_per_update = episode_length × n_rollout_threads
                       = 24 × 2
                       = 48

40 updates num_env_steps = 40 × 24 × 2 = 1920
```

因此相同40 updates时，2 Worker样本量是1 Worker的2倍。3 seeds的训练总采样量为5760 transitions，但每个seed仍独立运行和保存。

## N. 测试结果

所有命令均使用本地 `idc_ppo` Python、本地固定HARL/runtime；未联网、未安装依赖。核心正式性能命令模式：

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  train/train_harl_mappo_short.py `
  --updates 2 --rollout-threads <1|2|4> `
  --checkpoint-interval 1 --device cpu `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path .tmp_harl_runtime `
  --output-dir runs/part10_<candidate>
```

固定样本命令模式：

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  marl/tests/parallel_rollout_benchmark.py `
  --workers <1|2|4> --transitions 96 --seed 7110
```

| 测试文件/运行 | 结果 | warnings | pytest耗时/运行wall time |
| --- | --- | ---: | ---: |
| `test_parallel_rollout_sampling.py`完整核心集 | 18 passed | 9 | 63.19 s |
| 新增正式失败summary单项 | 1 passed、18 deselected | 9 | 13.12 s |
| `test_training_checkpoint_resume.py` | 15 passed | 1 | 7.28 s |
| `test_training_metrics_logger.py` | 13 passed | 17 | 10.79 s |
| `test_bess_effective_action_mask.py` | 12 passed | 801 | 112.52 s |
| `test_harl_mappo_formal_entry.py` | 9 passed | 257 | 45.59 s |
| `test_harl_mappo_smoke.py` | 1 passed | 202 | 34.95 s |
| `test_grid_cache_formal_entry.py` | 11 passed | 57 | 123.81 s |
| 1 Worker正式2-update | exit 0 | 无功能错误 | 81.963 s |
| 2 Worker正式2-update | exit 0 | 无功能错误 | 56.565 s |
| 4 Worker正式2-update | exit 0 | 无功能错误 | 88.329 s |
| 2 Worker parent 1-update + 新进程resume到2 | exit 0 / exit 0 | 无功能错误 | 29.202 s / 25.139 s |

合计80个最终有效断言通过、0失败。大量warning主要是pandapower既有 `tap_dependency_table` deprecation；另有pytest cache目录权限警告和原smoke tensor-to-scalar warning，不影响验收结果。

曾有一次回归收集命令因只设置项目`PYTHONPATH`而报 `ModuleNotFoundError: harl`，以及两次因使用了错误artifact环境变量名而全部skip；修正为本地HARL/runtime路径和正确checkpoint变量后同一测试15/15通过。这些是验收命令配置错误，不是产品断言失败。

最终残留进程检查：`PYTHON_PROCESS_COUNT=0`。

## O. 问题分类

### 阻塞多Worker正式训练

无。shape、seed、Reward、cache、日志、Mask、异常传播和checkpoint均已通过。

### 阻塞40-update前

无新的第十部分阻塞。开始前必须保留已解析配置快照，并确认实际值为2 Worker、48 transitions/update、1920 num_env_steps。

### 建议改进

- 如后续需要更完整性能画像，可增加只读采样器记录主/child平均CPU与峰值RSS；本次未为跑分引入额外依赖。
- 未来大规模Worker可研究CPU affinity或只读数据映射，但4 Worker当前无收益证据，不应扩展。
- pytest cache目录权限警告可在独立工程维护任务处理，不影响训练正确性。

### 可以保持现状

- 2 Worker、spawn、seed stride=1000。
- 每Worker独立OPF/MEF cache。
- 主进程单写日志/checkpoint。
- HARL标准buffer、GAE、actor/critic update流程。
- BESS有效动作Mask和有界Box动作实现。

## P. 第十部分判断

| 项目 | 判断 |
| --- | --- |
| 1-Worker正确性 | 通过 |
| 2-Worker正确性 | 通过 |
| 4-Worker正确性 | 通过 |
| 多Worker可复现性 | 2 Worker精确复现通过，最大训练差异0 |
| 多Worker Checkpoint恢复 | 2 Worker新进程精确恢复通过，最大训练差异0 |
| 性能最优Worker数 | 正式训练吞吐/稳定性综合为2 |
| 正式短训练推荐 | 2 Worker，可用于后续正式短训练 |

总体选择：**1. 通过，推荐Worker数已经验证并可用于正式训练。**

## Q. 最小后续建议

1. 保持 `configs/harl_mappo_short.yaml` 的 `n_rollout_threads: 2`。
2. 40-update入口继续使用 `train/train_harl_mappo_short.py`，每seed显式记录`updates=40`、`num_env_steps=1920`、worker seeds `[base, base+1000]`。
3. 后续MAPPO/HAPPO比较固定2 Worker、episode_length=24和相同总样本语义。
4. 不再修改并行采样、物理、Reward、Mask或checkpoint语义；下一阶段另行执行，不在本报告进入第十一部分。

