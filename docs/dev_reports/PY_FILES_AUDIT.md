# Python 文件资产盘点报告

## 1. 总体结论

本项目当前共有 15 个 `.py` 文件。核心建模代码集中在 `task.py`、`power_model.py`、`task_model.py`、`IDCPriceEnv20D_ultimate.py`，它们共同描述任务对象、服务器/IDC 功耗、任务生成与执行、PPO Gym 环境、BESS、碳排放和电网购电侧指标。`config_ultimate.py`、`data_loader.py`、`experiment_cases.py` 是训练和评估共享的配置/输入层。`train_ppo_ultimate.py` 是 PPO 训练入口，`eval_base.py` 是当前最重要的统一评估入口，`ga_base.py` 和 `pso_base.py` 是用于 PPO/启发式对比的 baseline 算法。`demo_random_env_test.py`、`demo_task_model.py` 属于连通性和建模验证 demo。`export_task_timeline_sin.py` 明显偏向报告图表/任务甘特图导出；`eval_nn_reuse.py` 是历史 GA/PSO 方案最近邻复用实验，更像阶段性实验脚本。若后续主线转向 MEF / IEEE 14 节点 / OPF / MAPPO，最需要保留并扩展的是配置层、数据层、环境层、任务/功耗模型、统一评估入口和训练入口；报告图表导出、单次 demo、历史方案复用脚本建议移出根目录或归档。

## 2. 文件总览表

| 文件名 | 类型分类 | 主要作用 | 是否为核心文件 | 是否可直接运行 | 被哪些文件导入 | 依赖哪些文件 | 产生哪些输出 | 当前建议 | 删除风险 |
|---|---|---|---|---|---|---|---|---|---|
| `config_ultimate.py` | 配置/数据输入层 | 集中保存环境、奖励、PPO、输出根目录和默认 seed 配置 | 是 | 否 | `demo_random_env_test.py`, `eval_base.py`, `eval_nn_reuse.py`, `experiment_cases.py`, `export_task_timeline_sin.py`, `ga_base.py`, `pso_base.py`, `train_ppo_ultimate.py` | 无 | 无；提供 `report_outputs` 路径解析 | 必须保留 | 高：删除会导致核心训练/环境/评估崩溃 |
| `data_loader.py` | 配置/数据输入层 | 读取可选 CSV 时序数据，构造环境外部输入序列 | 是 | 否 | `demo_random_env_test.py`, `eval_base.py`, `eval_nn_reuse.py`, `export_task_timeline_sin.py`, `ga_base.py`, `pso_base.py`, `train_ppo_ultimate.py` | 无 | 无；读取外部 CSV | 必须保留 | 高：删除会导致核心训练/环境/评估崩溃 |
| `demo_random_env_test.py` | demo/连通性测试 | 随机动作跑完一个 episode，检查环境 reset/step 和指标输出 | 否 | 是 | 无 | `IDCPriceEnv20D_ultimate.py`, `config_ultimate.py`, `data_loader.py` | 控制台日志 | 可移动到 tools/ 或 scripts/ | 低：删除主要影响 demo/报告辅助 |
| `demo_task_model.py` | demo/连通性测试 | 检查任务模型、功耗模型、价格感知规则调度和任务统计 | 否 | 是 | 无 | `task_model.py` | 控制台日志 | 可移动到 tools/ 或 scripts/ | 低：删除主要影响 demo/报告辅助 |
| `eval_base.py` | 评估入口 | 统一评估规则策略、PPO，并合并 GA/PSO CSV 结果 | 是 | 是 | 无 | `IDCPriceEnv20D_ultimate.py`, `config_ultimate.py`, `data_loader.py`, `experiment_cases.py` | `all_results.csv`, `summary.csv`, `hourly_result.csv`, `hourly_mean.csv` | 必须保留 | 高：删除会导致核心训练/环境/评估崩溃 |
| `eval_nn_reuse.py` | 临时实验脚本 | 用历史 GA/PSO 最优方案做最近邻迁移评估，并可对比 PPO | 否 | 是 | 无 | `config_ultimate.py`, `data_loader.py`, `IDCPriceEnv20D_ultimate.py` | `feature_config.csv`, `nearest_matches.csv`, `all_results.csv`, `summary.csv` | 可移动到 legacy/ | 中：删除会影响实验、复现实验或画图 |
| `experiment_cases.py` | 配置/数据输入层 | 管理 main/no_bess/carbon 权重实验配置，并输出关键元数据 | 是 | 否 | `eval_base.py`, `train_ppo_ultimate.py` | `config_ultimate.py` | 无；打印 case 摘要 | 必须保留 | 高：删除会导致核心训练/环境/评估崩溃 |
| `export_task_timeline_sin.py` | 图表/报告辅助 | 从 PPO 模型导出任务调度表、执行时间线和甘特图 | 否 | 是 | 无 | `IDCPriceEnv20D_ultimate.py`, `config_ultimate.py`, `data_loader.py` | `task_schedule.csv`, `task_execution_timeline.csv`, `task_gantt.png` | 可移动到 tools/ 或 scripts/ | 低：删除主要影响 demo/报告辅助 |
| `ga_base.py` | baseline 算法 | 遗传算法搜索 24 小时动作计划，生成 GA 对比结果 | 否 | 是 | 无 | `config_ultimate.py`, `data_loader.py`, `IDCPriceEnv20D_ultimate.py` | `ga_results.csv`, 可选 `ga_plan_seed*.npy` | 建议保留 | 中：删除会影响实验、复现实验或画图 |
| `IDCPriceEnv20D_ultimate.py` | 环境层 | Gymnasium 环境，连接任务调度、功耗、BESS、碳排放、奖励和观测 | 是 | 否 | `demo_random_env_test.py`, `eval_base.py`, `eval_nn_reuse.py`, `export_task_timeline_sin.py`, `ga_base.py`, `pso_base.py`, `train_ppo_ultimate.py` | `task_model.py` | 无；运行时返回 obs/reward/info | 必须保留 | 高：删除会导致核心训练/环境/评估崩溃 |
| `power_model.py` | 核心建模层 | IDC 服务器异构、IT 功耗、PUE、分时电价和阶段一指标计算 | 是 | 否 | `task_model.py` | 无 | 无 | 必须保留 | 高：删除会导致核心训练/环境/评估崩溃 |
| `pso_base.py` | baseline 算法 | 粒子群算法搜索 24 小时动作计划，生成 PSO 对比结果 | 否 | 是 | 无 | `config_ultimate.py`, `data_loader.py`, `IDCPriceEnv20D_ultimate.py` | `pso_results.csv`, 可选 `pso_plan_seed*.npy` | 建议保留 | 中：删除会影响实验、复现实验或画图 |
| `task.py` | 核心建模层 | 定义单个任务对象、任务状态、执行日志和剩余工作量更新 | 是 | 否 | `task_model.py` | 无 | 无 | 必须保留 | 高：删除会导致核心训练/环境/评估崩溃 |
| `task_model.py` | 核心建模层 | 任务 profile、任务生成、初始积压、任务执行仿真和任务指标 | 是 | 否 | `IDCPriceEnv20D_ultimate.py`, `demo_task_model.py` | `task.py`, `power_model.py` | 无 | 必须保留 | 高：删除会导致核心训练/环境/评估崩溃 |
| `train_ppo_ultimate.py` | 训练入口 | 创建 PPO 环境、训练模型、保存 checkpoint/best/final/metadata | 是 | 是 | 无 | `config_ultimate.py`, `data_loader.py`, `experiment_cases.py`, `IDCPriceEnv20D_ultimate.py` | PPO 模型、日志、`training_metadata.json` | 必须保留 | 高：删除会导致核心训练/环境/评估崩溃 |

## 3. 每个文件的详细分析

### `config_ultimate.py`

**一句话定位：**
当前项目的全局配置中心，负责把环境、奖励、PPO 和输出路径参数从训练/评估/环境代码中抽出来。

**主要内容：**
`OUTPUT_ROOT`、`resolve_output_path()`、`ENV_CONFIG`、`DATA_CONFIG`、`REWARD_CONFIG`、`PPO_CONFIG`、`DEFAULT_EVAL_SEED`。

**依赖关系：**
- import 本项目文件：无。
- 被本项目文件 import：`demo_random_env_test.py`、`eval_base.py`、`eval_nn_reuse.py`、`experiment_cases.py`、`export_task_timeline_sin.py`、`ga_base.py`、`pso_base.py`、`train_ppo_ultimate.py`。
- 外部依赖/目录：依赖 `pathlib`；输出根目录默认是 `report_outputs`；`DATA_CONFIG` 可指向价格、碳因子、温度、PV、WT CSV。

**和当前主线的关系：**
价值很高。后续 IEEE 14 / OPF / MEF / LMP / MAPPO 参数都应从这里进入，避免继续把电网参数写死在环境或评估脚本中。

**建议处理：**
保留但建议重构。

**理由：**
它被训练、评估、baseline、demo 全面依赖。后续建议增加 `GRID_CONFIG`、`OPF_CONFIG`、`EMISSION_CONFIG`、`MAPPO_CONFIG`，并逐步把当前 `ENV_CONFIG` 中的电网购电/BESS/碳参数拆得更清楚。

### `data_loader.py`

**一句话定位：**
外部时序数据入口，用 CSV 读取 price、carbon、temperature、PV、WT 并转换成环境可用数组。

**主要内容：**
`load_time_series_csv()`、`load_optional_time_series_csv()`、`build_external_series_from_config()`。

**依赖关系：**
- import 本项目文件：无。
- 被本项目文件 import：`demo_random_env_test.py`、`eval_base.py`、`eval_nn_reuse.py`、`export_task_timeline_sin.py`、`ga_base.py`、`pso_base.py`、`train_ppo_ultimate.py`。
- 外部依赖/目录：依赖 `csv`、`numpy`、`pathlib`；读取用户配置的 CSV 文件；PV/WT 缺省为全零。

**和当前主线的关系：**
价值很高。后续接入 IEEE 14 节点拓扑、负荷曲线、机组参数、线路参数、MEF/LMP 时序数据时，应扩展此层或拆出 `grid_model/ieee14_loader.py`。

**建议处理：**
保留但建议重构。

**理由：**
当前已是所有入口共享的数据适配层。建议避免把 IEEE 14 数据读取直接塞入环境文件，而是在数据层返回结构化 grid input，再由环境或 wrapper 使用。

### `demo_random_env_test.py`

**一句话定位：**
环境随机动作连通性 smoke test，用于确认 `IDCPriceEnv20D` 能完整跑完 24 小时 episode。

**主要内容：**
`main()` 创建环境、`reset()`、随机采样 action、循环 `step()`，打印每小时和最终指标。

**依赖关系：**
- import 本项目文件：`IDCPriceEnv20D_ultimate.py`、`config_ultimate.py`、`data_loader.py`。
- 被本项目文件 import：无。
- 外部依赖/目录：无文件输出，仅控制台输出。

**和当前主线的关系：**
对主线有辅助价值，适合在每次重构环境、观测维度、动作维度、BESS 或电网 wrapper 后快速检查。2026-07-27 已将旧 docstring 的“动作维度 22”修正为参数化的 N+3 描述；默认配置仍由训练脚本和运行时检查确认为 23 维。

**建议处理：**
移动到 scripts/。

**理由：**
它不是核心库文件，也不被其他文件引用，但作为连通性测试有保留价值。移动后可作为 `scripts/smoke_random_env.py` 一类工具。

### `demo_task_model.py`

**一句话定位：**
任务层和功耗层 demo，用规则负载计划验证任务生成、任务执行、功耗和阶段一成本指标。

**主要内容：**
`main()` 创建 `IDCEnergyTaskModel`，生成 demo tasks 和初始积压任务，构建价格感知计划，运行 `simulate_task_execution_stepwise()`，计算 PUE、成本、任务完成率并打印表格。

**依赖关系：**
- import 本项目文件：`task_model.py`。
- 被本项目文件 import：无。
- 外部依赖/目录：依赖 `numpy`；无文件输出，仅控制台输出。

**和当前主线的关系：**
对任务建模仍有参考价值，但它绕过了 PPO 环境和 BESS/电网购电逻辑，更像早期阶段一验证脚本。后续 MEF/OPF 主线中价值低于 `IDCPriceEnv20D_ultimate.py` 和 `eval_base.py`。

**建议处理：**
移动到 scripts/。

**理由：**
不是运行训练/评估必需文件，但能帮助人工理解任务模型和功耗模型，不建议直接删除。

### `eval_base.py`

**一句话定位：**
当前最重要的统一评估入口，用同一环境 seed 评估规则策略、PPO，并合并 GA/PSO 结果。

**主要内容：**
环境创建、指标抽取、小时级结果构造、`ZERO/ONE/RANDOM/FAST_NEUTRAL_BESS/UNIFORM_NEUTRAL_BESS/RULE_PRICE_ONLY/RULE_PRICE_BESS` 策略、PPO 模型解析与评估、GA/PSO CSV 导入、结果汇总与 CSV 输出、CLI `main()`。

**依赖关系：**
- import 本项目文件：`IDCPriceEnv20D_ultimate.py`、`config_ultimate.py`、`data_loader.py`、`experiment_cases.py`。
- 被本项目文件 import：无。
- 外部依赖/目录：依赖 `numpy`、`csv`、可选 `stable_baselines3`；读取 PPO `.zip`、GA/PSO CSV；输出到默认 `report_outputs/eval_<case>/`。

**和当前主线的关系：**
价值很高。后续加入 MEF、LMP、电压越限、线路负载率、OPF 可行性等指标时，最应该在这里统一汇总和比较。

**建议处理：**
保留但建议重构。

**理由：**
它承担论文/实验复现的评价主入口。建议未来拆分为 `evaluators/policies.py`、`evaluators/metrics.py`、`scripts/eval_base.py`，但当前不要移走，避免破坏既有运行方式。

### `eval_nn_reuse.py`

**一句话定位：**
阶段性实验脚本：用历史 GA/PSO 最优动作计划做场景特征最近邻匹配，并迁移到新 seed 上评估。

**主要内容：**
场景特征提取、特征组权重、标准化、加权欧式距离、最近邻匹配、历史动作计划加载、动作计划评估、PPO 对照评估、CSV 输出与 summary。

**依赖关系：**
- import 本项目文件：`config_ultimate.py`、`data_loader.py`、`IDCPriceEnv20D_ultimate.py`。
- 被本项目文件 import：无。
- 外部依赖/目录：依赖 `numpy`、`csv`；读取 `report_outputs/ga_out/ga_plan_seed*.npy`、`report_outputs/pso_out/pso_plan_seed*.npy`、默认 PPO 模型；输出 `feature_config.csv`、`nearest_matches.csv`、`all_results.csv`、`summary.csv`。

**和当前主线的关系：**
对主线不是基础能力，但对“历史方案复用/启发式迁移”有实验参考价值。它不直接帮助 IEEE 14/OPF/MEF 建模，也不适合作为 MAPPO 主线代码的一部分。

**建议处理：**
移动到 legacy/。

**理由：**
文件注释明确写着“方便写报告说明”，且依赖历史 GA/PSO `.npy` 方案库。建议归档到 `legacy/nn_reuse_experiments/`，保留复现实验价值，但从根目录移走。

### `experiment_cases.py`

**一句话定位：**
训练和评估共享的实验 case 配置管理器。

**主要内容：**
`get_experiment_case()` 支持 `main`、`no_bess`、`carbon_w0`、`carbon_w03`、`carbon_w05`；`key_env_config()`、`key_reward_config()`、`print_experiment_case()`。

**依赖关系：**
- import 本项目文件：`config_ultimate.py`。
- 被本项目文件 import：`eval_base.py`、`train_ppo_ultimate.py`。
- 外部依赖/目录：无文件输出，只打印 case 摘要。

**和当前主线的关系：**
有价值。后续可以扩展为 grid case / OPF case / emission case 管理，例如 `ieee14_base`、`ieee14_high_renewable`、`mef_dynamic`。

**建议处理：**
保留但建议重构。

**理由：**
虽然命名偏报告实验，但已经被训练和统一评估入口直接依赖。后续建议改名或扩展为更通用的 `experiment_registry.py`。

### `export_task_timeline_sin.py`

**一句话定位：**
报告辅助脚本，从指定 PPO 模型导出任务排序表、执行时间线 CSV 和甘特图。

**主要内容：**
`make_env()`、`run_ppo_and_export()`、`plot_gantt()`、CLI `main()`；读取 PPO 模型，运行一个 seed，遍历 `env.tasks` 生成任务表和时间线。

**依赖关系：**
- import 本项目文件：`IDCPriceEnv20D_ultimate.py`、`config_ultimate.py`、`data_loader.py`。
- 被本项目文件 import：无。
- 外部依赖/目录：依赖 `pandas`、`matplotlib`、`stable_baselines3`；读取默认 `ppo_outputs_sin/best_model/best_model.zip`；输出 `task_schedule.csv`、`task_execution_timeline.csv`、`task_gantt.png`。

**和当前主线的关系：**
对主线价值较低，主要服务报告可视化。后续如果研究需要展示 MAPPO 调度行为，可以保留思路，但不应留在根目录作为核心代码。

**建议处理：**
移动到 tools/。

**理由：**
文件名和默认模型路径都绑定 `sin` 版本，输出是任务甘特图和表格，明显属于图表/报告辅助。

### `ga_base.py`

**一句话定位：**
GA baseline，搜索完整 24 小时、23 维动作计划，用于和 PPO/规则策略比较。

**主要内容：**
`GAConfig`、`make_env()`、`evaluate_plan()`、`fitness_from_info()`、价格感知初始个体、种群初始化、锦标赛选择、交叉、变异、`run_ga()`、CSV 写入、CLI `main()`。

**依赖关系：**
- import 本项目文件：`config_ultimate.py`、`data_loader.py`、`IDCPriceEnv20D_ultimate.py`。
- 被本项目文件 import：无。
- 外部依赖/目录：依赖 `numpy`、`csv`；输出默认 `report_outputs/ga_out/ga_results.csv`，可选 `ga_plan_seed*.npy`。

**和当前主线的关系：**
对 PPO/未来 MAPPO 对比实验有价值。它不是核心建模层，但可作为优化上界或启发式 baseline。

**建议处理：**
保留在根目录。

**理由：**
当前论文/实验复现可能依赖 GA/PSO/PPO 横向比较。等项目结构稳定后，可移动到 `baselines/ga_base.py`。

### `IDCPriceEnv20D_ultimate.py`

**一句话定位：**
项目核心 Gymnasium 环境，负责把任务模型、服务器功耗、BESS、购电、碳排放、奖励函数和观测空间接到 RL 算法。

**主要内容：**
`IDCPriceEnv20D` 类；`__init__()` 按 N+3 配置动作空间，并按 6+10+6*N+6*horizon 配置底层观测空间；当前默认值为 23 维动作、280 维底层 observation，`GridCoupledEnv` 再追加 8 维。`reset()` 生成任务和初始化 episode；`step()` 处理动作、任务执行、BESS 充放电、电网购电、成本/碳/峰值/reward/info；辅助函数覆盖任务激活、任务选择、暂停恢复、deadline miss、SLA、任务池特征、服务器特征、完整 horizon 前瞻特征和观测构造。

**依赖关系：**
- import 本项目文件：`task_model.py`。
- 被本项目文件 import：`demo_random_env_test.py`、`eval_base.py`、`eval_nn_reuse.py`、`export_task_timeline_sin.py`、`ga_base.py`、`pso_base.py`、`train_ppo_ultimate.py`。
- 外部依赖/目录：依赖 `numpy`、`gymnasium`；可接收 price/carbon/T/PV/WT 时序；不直接写文件。

**和当前主线的关系：**
价值最高。后续 IEEE 14 / OPF / MEF / LMP 都会影响环境的状态、动作约束、reward 和 info 指标。不过不建议把 OPF 求解器直接写进这个大文件。

**建议处理：**
保留但建议重构。

**理由：**
它是所有训练、评估、baseline 的运行核心。后续建议通过 `grid_wrapper` 或 `GridCoupledEnv` 接入 OPF/MEF，避免环境文件继续膨胀。

### `power_model.py`

**一句话定位：**
IDC 电力/功耗基础模型，描述服务器异构、IT 功耗、PUE、分时电价和阶段一指标。

**主要内容：**
`IDCPowerModel` 类；`calc_it_power()`、`load_balance()`、`calc_pue_and_total_power()`、`create_price_curve()`、`evaluate_stage1_metrics()`。

**依赖关系：**
- import 本项目文件：无。
- 被本项目文件 import：`task_model.py`。
- 外部依赖/目录：依赖 `numpy`；不写文件。

**和当前主线的关系：**
对 IDC 侧建模非常重要，但电网侧真实性目前不足。后续不应把 IEEE 14 OPF 塞进此文件，而应让它继续负责 IDC 内部功耗，电网侧由新模块负责。

**建议处理：**
保留但建议重构。

**理由：**
它是 `task_model.py` 的父类，删除会使环境链断裂。未来可拆为 `models/idc_power.py`，并与 grid 模块通过 IDC 购电功率接口交互。

### `pso_base.py`

**一句话定位：**
PSO baseline，搜索完整 24 小时、23 维动作计划，用于和 PPO/GA/规则策略比较。

**主要内容：**
`PSOConfig`、`make_env()`、`fitness_from_info()`、`evaluate_plan()`、价格感知初始粒子、粒子初始化、`run_pso()`、CSV 写入、CLI `main()`。

**依赖关系：**
- import 本项目文件：`config_ultimate.py`、`data_loader.py`、`IDCPriceEnv20D_ultimate.py`。
- 被本项目文件 import：无。
- 外部依赖/目录：依赖 `numpy`、`csv`；输出默认 `report_outputs/pso_out/pso_results.csv`，可选 `pso_plan_seed*.npy`。

**和当前主线的关系：**
对 baseline 对照有价值，但不是未来 MAPPO/CTDE 主干。保留可帮助比较集中式启发式搜索与 RL 的效果差异。

**建议处理：**
保留在根目录。

**理由：**
当前 `eval_base.py` 会自动导入 PSO CSV，说明它仍服务于当前实验闭环。后续可和 GA 一起移动到 `baselines/`。

### `task.py`

**一句话定位：**
任务实体定义文件，用 `Task` dataclass 统一管理任务属性、状态和执行日志。

**主要内容：**
`Task` dataclass；属性包括任务类型、到达、持续时间、负载曲线、工作量、deadline、priority、可中断/可并行；方法包括 `latest_finish_time`、`is_finished`、`avg_load`、`execute()`。

**依赖关系：**
- import 本项目文件：无。
- 被本项目文件 import：`task_model.py`。
- 外部依赖/目录：依赖 `numpy`、`dataclasses`；不写文件。

**和当前主线的关系：**
价值很高。MAPPO+CTDE+注意力机制很可能需要更丰富的任务对象特征或任务池编码，`Task` 是自然扩展点。

**建议处理：**
保留在根目录。

**理由：**
它是任务建模的最小实体，删除会导致 `task_model.py` 和环境链断裂。后续可增加数据中心/节点归属、任务可迁移性、服务等级等字段。

### `task_model.py`

**一句话定位：**
任务生成与执行模型，继承功耗模型并生成任务 profile、任务到达、初始积压和任务级指标。

**主要内容：**
`IDCEnergyTaskModel` 类；任务 profile 初始化、任务参数采样、随机/demo 任务生成、初始积压任务、到达曲线、价格感知计划、任务负载曲线、调度约束检查、队列仿真、任务执行仿真。

**依赖关系：**
- import 本项目文件：`task.py`、`power_model.py`。
- 被本项目文件 import：`IDCPriceEnv20D_ultimate.py`、`demo_task_model.py`。
- 外部依赖/目录：依赖 `numpy`；不写文件。

**和当前主线的关系：**
价值很高。未来 MAPPO 中任务池注意力、任务选择动作、任务迁移和多智能体分配都需要从这里或其重构版本出发。

**建议处理：**
保留但建议重构。

**理由：**
当前同时承担任务生成、任务执行仿真和继承功耗模型职责，长期看应拆成 `task_generator.py`、`task_execution.py`、`idc_power_model.py`，但当前不可删除。

### `train_ppo_ultimate.py`

**一句话定位：**
PPO 训练入口，基于指定实验 case 创建环境并训练/保存模型。

**主要内容：**
`make_env()`、`sanity_check_env()`、`save_training_metadata()`、CLI `main()`；创建 `Monitor` 环境、配置 `CheckpointCallback` 和 `EvalCallback`、训练 Stable-Baselines3 PPO、保存 final/best model 和 metadata。

**依赖关系：**
- import 本项目文件：`config_ultimate.py`、`data_loader.py`、`experiment_cases.py`、`IDCPriceEnv20D_ultimate.py`。
- 被本项目文件 import：无。
- 外部依赖/目录：依赖 `stable_baselines3`；输出 `report_outputs/ppo_outputs_<run_name>_<case>/models`、`logs`、`best_model`、`training_metadata.json`。

**和当前主线的关系：**
当前 PPO 主训练入口，必须保留。后续 MAPPO+CTDE 会新增训练入口，但此文件仍可作为单智能体 PPO baseline。

**建议处理：**
保留在根目录。

**理由：**
删除会导致当前项目无法训练 PPO baseline。未来可复制其配置/metadata/seed 管理模式到 `train_mappo.py`。

## 4. 依赖关系图

核心建模链：

```text
IDCPriceEnv20D_ultimate.py
→ task_model.py
→ task.py
→ power_model.py
```

PPO 训练链：

```text
train_ppo_ultimate.py
→ experiment_cases.py
→ config_ultimate.py
→ data_loader.py
→ IDCPriceEnv20D_ultimate.py
→ task_model.py
→ task.py / power_model.py
```

统一评估链：

```text
eval_base.py
→ experiment_cases.py
→ config_ultimate.py
→ data_loader.py
→ IDCPriceEnv20D_ultimate.py
→ task_model.py
→ task.py / power_model.py
```

GA / PSO baseline 链：

```text
ga_base.py / pso_base.py
→ config_ultimate.py
→ data_loader.py
→ IDCPriceEnv20D_ultimate.py
→ task_model.py
→ task.py / power_model.py
```

demo / 报告辅助链：

```text
demo_random_env_test.py
→ config_ultimate.py
→ data_loader.py
→ IDCPriceEnv20D_ultimate.py
```

```text
demo_task_model.py
→ task_model.py
→ task.py / power_model.py
```

```text
export_task_timeline_sin.py
→ config_ultimate.py
→ data_loader.py
→ IDCPriceEnv20D_ultimate.py
→ PPO model zip
```

```text
eval_nn_reuse.py
→ config_ultimate.py
→ data_loader.py
→ IDCPriceEnv20D_ultimate.py
→ ga_plan_seed*.npy / pso_plan_seed*.npy / PPO model zip
```

## 5. 建议保留清单

### A. 必须保留的核心文件

- `config_ultimate.py`
- `data_loader.py`
- `IDCPriceEnv20D_ultimate.py`
- `task.py`
- `task_model.py`
- `power_model.py`
- `train_ppo_ultimate.py`
- `eval_base.py`
- `experiment_cases.py`

### B. 建议保留的实验文件

- `ga_base.py`
- `pso_base.py`

这两个文件对 PPO/GA/PSO 对比、启发式上界和论文复现实验仍有价值。后续可移动到 `baselines/`，但不建议删除。

### C. 建议移动到 tools/ 或 scripts/ 的辅助文件

- `demo_random_env_test.py`
- `demo_task_model.py`
- `export_task_timeline_sin.py`

这些文件不是核心依赖，但可用于环境连通性检查、任务模型解释或报告图表导出。

### D. 建议移动到 legacy/ 的旧文件

- `eval_nn_reuse.py`

该文件依赖历史 GA/PSO 方案库并服务最近邻复用实验。建议归档保留，避免它和主线评估入口混在根目录。

### E. 疑似可删除候选

当前没有建议立即删除的 `.py` 文件。`export_task_timeline_sin.py` 和 `eval_nn_reuse.py` 更像报告/阶段性实验代码，但仍可能用于复现已有结果，因此只标记为移动或归档候选，不标记为可直接删除。

## 6. 与节能减排大赛相关的临时代码识别

更像节能减排报告、图表、摘要、实验展示临时加入的文件：

| 文件名 | 判断 | 是否影响当前研究主线 | 是否建议从根目录移走 | 建议目录 |
|---|---|---|---|---|
| `export_task_timeline_sin.py` | 默认模型路径和输出名绑定 `sin` 版本，核心功能是任务表和甘特图导出 | 不影响核心训练/环境；只影响报告可视化 | 是 | `report_tools/` 或 `legacy/report_competition/` |
| `eval_nn_reuse.py` | 注释写明保存特征配置方便写报告，依赖历史 GA/PSO 方案库 | 对主线不是必需，但影响历史迁移实验复现 | 是 | `legacy/nn_reuse_experiments/` 或 `legacy/report_competition/` |
| `demo_task_model.py` | 大量打印阶段一成本、PUE、任务表，适合展示建模过程 | 不影响训练/评估主链 | 是 | `scripts/` 或 `report_tools/` |
| `demo_random_env_test.py` | 更像 smoke test，不是报告专用 | 对主线有连通性辅助价值 | 是 | `scripts/` |
| `experiment_cases.py` | 名称和注释偏 report training/evaluation，但已被训练/评估入口依赖 | 影响当前训练/评估 | 暂不建议移走 | 保留，未来重构为 `experiment_registry.py` |

## 7. 为后续 IEEE 14 节点 / OPF / MEF 改造的建议

建议新增目录结构：

```text
grid_model/
    __init__.py
    ieee14_loader.py
    grid_case.py
    opf_solver.py
    emission_model.py
    mef_calculator.py
    lmp_calculator.py
    grid_metrics.py

env_wrappers/
    grid_coupled_env.py

baselines/
    ga_base.py
    pso_base.py

scripts/
    demo_random_env_test.py
    demo_task_model.py
    export_task_timeline.py
```

具体交互建议：

- `config_ultimate.py` 应新增 `GRID_CONFIG`，包括 IEEE 14 case 路径、slack bus、IDC 接入节点、线路容量、机组成本、碳因子、OPF 求解开关、LMP/MEF reward 权重。
- `data_loader.py` 不宜只扩 CSV 时序；建议保留时序读取，同时新增或调用 `grid_model/ieee14_loader.py` 读取 bus/branch/gen/load 数据。
- `IDCPriceEnv20D_ultimate.py` 不建议直接写 OPF 求解器。更好的方式是 `IDCPriceEnv20D` 输出 IDC 购电功率需求，`env_wrappers/grid_coupled_env.py` 调用 `opf_solver.py`，再把 LMP、MEF、电压越限、线路负载率、OPF infeasible 等信息注入 reward/info/obs。
- `eval_base.py` 应新增电网真实性指标：`lmp_at_idc_bus`、`mef_kg_per_kwh`、`opf_feasible`、`voltage_violation_count`、`max_voltage_deviation`、`line_loading_max`、`line_overload_count`、`grid_congestion_cost`、`marginal_carbon_emission`。
- `train_ppo_ultimate.py` 可保留为单智能体 PPO baseline；MAPPO+CTDE 建议新增 `train_mappo.py`，不要在旧 PPO 文件上硬改。
- `task_model.py` 是未来注意力机制的任务池特征来源；建议后续把任务池编码从环境中抽到单独模块，便于 PPO/MAPPO 共享。

## 8. 最终人工决策表

| 文件名 | 你的建议 | 我是否删除/移动/保留 |
|---|---|---|
| `config_ultimate.py` | 保留但建议重构 | |
| `data_loader.py` | 保留但建议重构 | |
| `demo_random_env_test.py` | 移动到 scripts/ | |
| `demo_task_model.py` | 移动到 scripts/ | |
| `eval_base.py` | 保留但建议重构 | |
| `eval_nn_reuse.py` | 移动到 legacy/ | |
| `experiment_cases.py` | 保留但建议重构 | |
| `export_task_timeline_sin.py` | 移动到 tools/ 或 report_tools/ | |
| `ga_base.py` | 保留，后续可移动到 baselines/ | |
| `IDCPriceEnv20D_ultimate.py` | 保留但建议重构 | |
| `power_model.py` | 保留但建议重构 | |
| `pso_base.py` | 保留，后续可移动到 baselines/ | |
| `task.py` | 保留在根目录 | |
| `task_model.py` | 保留但建议重构 | |
| `train_ppo_ultimate.py` | 保留在根目录 | |
