# MAPPO短训练前检查——第十一部分：固定评估体系、确定性推理与公平比较

检查日期：2026-07-28  
项目：`C:\Users\bulio\Desktop\IDC\ultimate_simplify`  
HARL：`C:\Users\bulio\Desktop\IDC\HARL`  
正式入口：`eval/eval_harl_mappo_fixed.py`

## 结论摘要

第十一部分通过。仓库现已具备不依赖pytest、旧PPO或训练Runner临时对象的正式双Agent固定评估入口。`smoke_v1`已物化为3个显式环境状态快照；已有2-update MAPPO模型已完成3场景、72步确定性评估，0失败。run/model-only Actor与完整checkpoint Actor的固定输入和场景2026完整24步轨迹逐值一致，最大差异0。

本部分未运行40-update训练，未修改物理环境、Reward、MAPPO loss、GAE、BESS Mask、Grid/Safe目标、网络、并行采样或checkpoint恢复语义，也未进入第十二部分。

### 核心问题总览

| 项目 | 结论 |
| --- | --- |
| 旧脚本是否匹配双Agent | 否，保持为legacy single-agent evaluation |
| 正式双Agent固定入口 | 已新增 |
| 默认确定性动作 | 是；正式入口拒绝`deterministic=false` |
| `eval()`/`no_grad()` | 均执行并验收 |
| IDC/BESS Actor加载 | 两个Actor分别严格加载，顺序固定为`[idc,bess]` |
| BESS物理语义 | 22维输出，Bridge只切第0维；原映射`2*x-1`保持不变 |
| 是否恢复训练环境 | 否；checkpoint只提取两个Actor和兼容信息 |
| 场景固定程度 | 任务、服务器、RNG、SOC、cache、外部曲线和初始obs/state均物化 |
| 同模型同场景复现 | 完整24步最大差异0 |
| 跨模型环境共享 | 无；每个model×scenario均fresh环境 |
| Reward/物理指标 | 复用第八部分canonical语义并重构通过 |
| MAPPO/HAPPO/HGTA共用 | Suite/Evaluator/Schema可共用；当前loader只正式支持MAPPO |
| model-only/checkpoint一致 | Actor输出和固定轨迹最大差异0 |
| 逐场景配对比较 | 已支持，episode CSV保留model×scenario和双方hash |
| 5-update门槛 | 已具备；正式结果前仍须冻结独立Validation Suite |

## A. 旧评估脚本审计

| 文件 | 原用途 | 双Agent兼容 | 可复用 | 必须隔离 |
| --- | --- | --- | --- | --- |
| `eval_base.py` | 根目录兼容启动器 | 否 | 启动器形式 | 不能作为正式入口 |
| `eval/eval_base.py` | 规则/RANDOM/GA/PSO/SB3 PPO综合评估 | 否 | 历史物理字段说明 | 单flat动作、单SB3 Actor、`best_model.zip`、随机基线、普通reset seed |
| `legacy/nn_reuse_experiments/eval_nn_reuse.py` | GA/PSO计划最近邻复用及SB3 PPO | 否 | 历史实验 | 直接构造旧单Agent环境、绕过Grid、单Actor、plan clip/补维 |
| `scripts/trace_grid_coupled_episode.py` | 单Agent物理trace | 否 | legacy诊断 | 不属于HARL固定评估 |

`eval/eval_base.py::make_env()`仅构造`IDCPriceEnv20D → GridCoupledEnv`；`evaluate_ppo()`只加载一个SB3模型。`eval_nn_reuse.py::make_env()`甚至直接构造旧`IDCPriceEnv20D`。两者即便调用`deterministic=True`，也不能加载当前两个HARL Actor、294维状态或BESS padding接口。

正式新入口静态测试确认不import `eval_base`、`eval_nn_reuse`、`stable_baselines3`或legacy模块。旧代码明确保留为legacy，没有堆入双Agent分支。

## B. 修改文件

新增：

- `configs/harl_mappo_fixed_eval.yaml`：Smoke/Validation/Test配置。
- `eval/eval_harl_mappo_fixed.py`：正式CLI。
- `marl/evaluation/fixed_scenario_suite.py`：快照生成、完整性及兼容验证。
- `marl/evaluation/model_loader.py`：run/checkpoint/model-only双Actor加载。
- `marl/evaluation/evaluator.py`：确定性单Worker评估。
- `marl/evaluation/metrics.py`：step/episode/aggregate持久化。
- `marl/evaluation/__init__.py`、`marl/tests/test_fixed_evaluation.py`。
- `evaluation_suites/smoke_v1/`：3个`.pt`、3个scenario manifest和suite manifest。
- 本报告。

修改：

- `marl/logging/training_metrics.py`：新增无文件副作用的`CanonicalMetricBuilder`，复用第八部分字段提取与聚合。
- `marl/logging/__init__.py`：导出builder。

保持不动：

- 两个legacy评估脚本。
- 训练入口、2 Worker采样、物理/Reward/OPF/MEF/cache规则。
- HARL Actor/Critic、loss、GAE、bounded Box、BESS Mask和训练resume语义。

## C. 固定场景Suite

| Suite | 类型 | seeds | 已物化 | 用途 |
| --- | --- | --- | --- | --- |
| `smoke_v1` | smoke | 2026, 2027, 2028 | 是 | 工程正确性，不具统计充分性 |
| `validation_v1` | validation | 12026～12030 | 否 | checkpoint/超参数选择 |
| `test_v1` | test | 22026～22030 | 否 | 最终结果，禁止反向调参 |

固定协议：`n_eval_workers=1`、CPU、`deterministic=true`、episode length=24。训练仍独立保持2 Worker；评估步数不由训练Worker数推导。

最终Suite hash：

```text
b74632bcb33a8fc00f682668e1677307a03376b0c2188f2afc29fdcb4d80593a
```

## D. 场景快照与Manifest

生成流程：

```text
fresh单环境(seed)
→ reset并完成初始Grid/OPF/MEF
→ 保存initial obs/state
→ single_environment_state_dict()
→ 保存外部曲线、配置fingerprint和版本
→ 内容hash、文件hash、size和manifest
→ close
```

快照保存显式字段而非整个环境对象，包含任务及动态状态、task/server RNG、服务器参数、BESS SOC/能量、OPF/MEF cache entries与计数、Bridge/Padding状态、初始`(2,288)` obs、`(2,294)` state，以及price、carbon、PV、lambda、Grid load scale和USEP曲线。当前环境没有温度曲线，因此明确记录`temperature_available=false`，未伪造数据。

加载会验证suite/scenario schema、文件存在、size、文件SHA-256、canonical content hash、场景数量/ID/seed、当前Grid CSV hash、HARL HEAD、环境/Reward/data fingerprint及所有维度。损坏场景测试已确认fail-fast。

评估恢复严格为：fresh环境 → 不reset → 恢复显式状态 → 使用快照initial obs/state → 从hour 0执行。不会在恢复后再次reset，不跨场景/模型继承SOC或cache。

## E. 模型加载

支持：

1. `--run-dir`：读取resolved config、run metadata和两个model-only Actor文件。
2. `--checkpoint`：只提取两个Actor state、compatibility、provenance和saved update。
3. `--model-dir`：读取两个Actor文件，但必须找到相邻resolved config或显式提供`--model-config`；不猜默认网络。

checkpoint评估不会恢复optimizer、Critic、ValueNorm、训练RNG、训练环境/cache、buffer或logger。Critic不参与动作选择，`critic_metrics_available=false`。

兼容检查覆盖algorithm、Agent顺序、obs/state/padded/effective动作维度、padding策略、bounded Box、hidden sizes、activation、feature normalization、RNN flags、experiment case和环境/Reward/data fingerprint。错误algorithm、顺序、obs/action/effective维度及Suite fingerprint均已验证会拒绝。

验收模型：训练seed 7110、update 2；model-only pair hash为：

```text
8374247c614f9e695a28bc917b89c2d288176ac843afc5dcd4b50cd1c18e7706
```

## F. 确定性推理

实际调用：

```python
actor.actor.eval()
with torch.no_grad():
    action, next_rnn = actor.act(..., deterministic=True)
```

Actor optimizer在加载后置为`None`；模型只加载一次。每个场景RNN state清零、mask置1，terminal后mask置0且不复用于下一场景。评估前后参数hash一致，所有`.grad is None`，Actor保持eval mode。

正式入口不提供隐式随机评估；`deterministic=False`直接拒绝。同一模型、同一场景2026的run/model-only与checkpoint 24步CSV完全一致，训练相关最大差异0。

## G. BESS动作与双Agent语义

保持既有链：

```text
BESS Actor输出22维
→ HarlPaddedBridge只切dim 0
→ 原多Agent动作链
→ signed physical request = 2 * padded_dim0 - 1
```

日志记录`bess_action_padded_dim0`、`bess_action_physical`及virtual mean/std/min/max。没有手工清零虚拟维、求均值、clip或重写映射。相同dim0、不同虚拟21维的两个fresh环境下一步transition最大差异0。

首次Smoke曾在第一步失败：初版错误要求physical数值等于归一化dim0。审计确认原映射是`2*x-1`后，只修正评估校验，没有改动作链；最终v3为0失败。

## H. 评估指标与输出Schema

正式输出：

```text
evaluations/part11_smoke_acceptance_v3/
├── evaluation_manifest.json
├── step_metrics.csv
├── episode_metrics.csv
├── aggregate_metrics.csv
├── aggregate_metrics.json
└── logs/
```

Step每行对应model×scenario×hour，包含suite/model/scenario hash、训练seed/update、IDC动作统计、BESS动作诊断及第八部分全部Reward/任务/成本/碳/BESS/Grid/cache/Safe字段。

Episode每行对应model×scenario，保留status/failure type/message；失败不会静默删除。Aggregate对24项指标输出mean、sample std、median、min、max及成功/失败场景数，并标记Smoke为`engineering smoke only; not statistically sufficient`。本部分未做统计显著性检验。

最终Schema：72 step行、3 episode行、24 aggregate行；每场景24步且terminal恰好一次，关键字段finite。

## I. 顺序独立性与环境隔离

Evaluator循环为`scenario → model → fresh env → restore → episode → close`。场景文件按正序/反序加载后按ID对齐的content hash一致；model-only/checkpoint加载顺序反转后固定输入动作一致。

场景2026先由run模型评估，再由checkpoint模型在另一个fresh环境重复，初始/最终SOC、cache、任务指标和完整24步轨迹一致。为遵守“正式3场景+最多额外重复一个场景”，没有再运行完整反序3场景；顺序独立性由fresh restore、反序加载及单场景跨来源精确重复共同验证。

## J. Validation/Test隔离

配置和manifest显式区分`smoke/validation/test`：Smoke只做工程检查；Validation用于checkpoint与超参数选择；Test只用于最终结果。Test不得用于调Reward、学习率、HGTA或Safe阈值。

本部分只物化Smoke。Validation/Test seed空间和用途已隔离但未执行，符合任务范围。当前没有“训练Reward最高即best model”逻辑；正式入口支持显式传入多个checkpoint，主选择指标需在正式实验设计中预先确定。

## K. 测试结果

Suite生成命令：

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  -m eval.eval_harl_mappo_fixed `
  --generate-suite --overwrite-suite --generate-suite-only `
  --suite-id smoke_v1
```

正式评估命令模式：

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  -m eval.eval_harl_mappo_fixed `
  --suite-id smoke_v1 `
  --suite-dir evaluation_suites/smoke_v1 `
  --run-dir <part10-2worker-update2-run> `
  --output-dir evaluations/part11_smoke_acceptance_v3 `
  --evaluation-id part11-smoke-acceptance-v3
```

额外等价测试使用同一入口，加`--scenario-id scenario_000_seed_02026 --checkpoint <update_000002.pt>`。

| 验证 | 结果 | warnings | 耗时 |
| --- | --- | ---: | ---: |
| 最终Suite生成/加载 | exit 0 | 0 | 13.8 s wall |
| 1模型×3场景Smoke | exit 0，72步，0失败 | 命令无warning输出 | 86.0 s wall；内部78.30 s |
| checkpoint额外重复2026 | exit 0，24步，0失败 | 命令无warning输出 | 35.9 s wall；内部27.37 s |
| `test_fixed_evaluation.py` | 21 passed | 49 | 20.84 s |
| fixed + 既有logger最终回归 | 34 passed | 65 | 30.48 s |

最终34项为固定评估21项和既有logger 13项，0 failed。warning主要是pandapower弃用警告和pytest cache权限警告。最终残留进程：`PYTHON_PROCESS_COUNT=0`。

## L. 正式Smoke Suite验收

以下仅为工程Smoke结果，不评价算法优劣或收敛。

| seed | Episode Reward | completion | final backlog | cost | carbon kg | peak kW | final SOC | OPF | MEF | safe violation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | -16.689471 | 0.449636 | 433722.875 | 14440.917 | 14048.124 | 1075.330 | 0.584151 | 1.0 | 1.0 | 0.0 |
| 2027 | -16.489490 | 0.425988 | 492204.530 | 14196.146 | 13829.925 | 1060.541 | 0.583412 | 1.0 | 1.0 | 0.0 |
| 2028 | -11.849721 | 0.567891 | 279258.925 | 13728.048 | 13412.694 | 1032.460 | 0.581881 | 1.0 | 1.0 | 0.0 |

验证同时确认：step Reward和episode Reward一致；非alias Reward分量重构误差满足`1e-6`；模型参数不变；BESS dim0映射正确；虚拟维不影响transition；suite/model hash均已记录。

## M. 性能与文件大小

| 项目 | 数值 |
| --- | ---: |
| 2026/2027/2028场景加载 | 0.00949 / 0.00792 / 0.00797 s |
| 三个24步episode | 24.102 / 25.462 / 24.659 s |
| 内部总评估时间 | 78.300 s |
| Smoke Suite大小 | 119,684 bytes |
| 正式评估输出大小 | 168,089 bytes |

模型不在每step重载，场景文件/hash不在每step读取或计算，CSV持续打开至评估结束。

## N. 问题分类

### 阻塞MAPPO短训练门槛

无。确定性加载、场景复现、Reward重构、BESS语义、fresh环境和Actor不变性均通过。

### 正式40-update结果分析前必须完成

流程上需物化并冻结`validation_v1`，预先确定checkpoint主选择指标；模型选择冻结后才运行`test_v1`。当前无代码阻塞。

### 建议改进

- 后续扩展Validation/Test场景数并预先登记统计协议。
- 可增加bootstrap、配对检验和论文表格生成，但不进入Smoke门槛。
- 未来HAPPO/HGTA只新增模型loader adapter，保持同一Suite/Evaluator/Schema。
- 若未来真正提供温度曲线，应升级Suite schema，不悄然覆盖`smoke_v1`。

### 可以保持现状

- 单Worker CPU顺序评估、显式状态快照、Actor-only checkpoint读取。
- 第八部分canonical指标、逐场景配对输出及Smoke/Validation/Test隔离。

## O. 第十一部分判断

| 项目 | 判断 |
| --- | --- |
| 固定场景正确性 | 通过；同seed和round-trip最大差异0 |
| 确定性模型推理 | 通过；重复24步最大差异0 |
| 模型加载兼容性 | 通过；run/model-only/checkpoint均验证 |
| BESS评估语义 | 通过；dim0有效、虚拟21维无物理影响 |
| Reward/物理指标一致性 | 通过 |
| 模型与场景顺序独立性 | 通过；fresh restore与顺序测试 |
| Validation/Test隔离 | 通过配置/协议隔离；按范围未物化 |
| Smoke Suite完整性 | 通过；72/3/24行，0失败，hash完整 |

总体选择：**1. 通过，固定评估体系可靠，可用于后续短训练门槛。**

## P. 最小后续建议

1. 5-update门槛后直接用同一`smoke_v1`评估对应checkpoint，不重新生成Suite。
2. 确认suite hash保持`b74632...593a`；不兼容变化应创建新版本，不覆盖历史Suite。
3. 正式40-update前物化`validation_v1`并预先定义主选择指标。
4. Test Suite只在模型选择冻结后使用一次。
5. MAPPO/HAPPO比较固定训练2 Worker、评估1 Worker，并使用同一Suite hash和Schema。
6. 本部分到此停止，不进入第十二部分。

