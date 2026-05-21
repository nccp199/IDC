# 代码轻量整理报告

## 1. 本次移动的文件

| 原路径 | 新路径 | 处理理由 |
|---|---|---|
| `demo_random_env_test.py` | `scripts/demo_random_env_test.py` | 环境随机动作连通性测试，不属于核心建模/训练/评估主链，适合放入脚本目录。 |
| `demo_task_model.py` | `scripts/demo_task_model.py` | 任务模型和功耗模型 demo，主要用于人工检查和展示，适合放入脚本目录。 |
| `export_task_timeline_sin.py` | `report_tools/export_task_timeline_sin.py` | 任务时序 CSV 和甘特图导出工具，明显偏报告/图表辅助。 |
| `eval_nn_reuse.py` | `legacy/nn_reuse_experiments/eval_nn_reuse.py` | 历史 GA/PSO 方案最近邻复用实验，属于阶段性实验脚本，建议归档到 legacy。 |

同时新增包标记文件：

- `scripts/__init__.py`
- `report_tools/__init__.py`
- `legacy/__init__.py`
- `legacy/nn_reuse_experiments/__init__.py`

## 2. 保留在根目录的核心文件

| 文件名 | 分类 | 说明 |
|---|---|---|
| `config_ultimate.py` | 配置/数据输入层 | 保存环境、奖励、PPO 参数和输出路径配置。 |
| `data_loader.py` | 配置/数据输入层 | 读取外部 price/carbon/temperature/PV/WT 时序 CSV。 |
| `experiment_cases.py` | 配置/数据输入层 | 管理 main/no_bess/carbon 权重等实验 case。 |
| `task.py` | 核心建模层 | 定义单个任务对象、状态和执行日志。 |
| `task_model.py` | 核心建模层 | 负责任务 profile、任务生成、任务执行仿真和任务指标。 |
| `power_model.py` | 核心建模层 | 负责服务器异构、IT 功耗、PUE 和阶段一成本指标。 |
| `IDCPriceEnv20D_ultimate.py` | 环境层 | Gymnasium 环境，连接任务调度、功耗、BESS、成本、碳排和 reward。 |
| `train_ppo_ultimate.py` | 训练入口 | Stable-Baselines3 PPO 训练入口。 |
| `eval_base.py` | 评估入口 | 统一评估规则策略、PPO，并合并 GA/PSO CSV。 |
| `ga_base.py` | baseline 算法 | GA 搜索 24 小时动作计划。 |
| `pso_base.py` | baseline 算法 | PSO 搜索 24 小时动作计划。 |

## 3. import 修复说明

四个被移动脚本都加入了相同的项目根目录自动定位逻辑：

- 从 `Path(__file__).resolve().parents` 向上查找；
- 找到包含 `config_ultimate.py` 或 `IDCPriceEnv20D_ultimate.py` 的目录后视为 `PROJECT_ROOT`；
- 将 `PROJECT_ROOT` 插入 `sys.path`；
- 再导入根目录中的项目模块。

这种方式不依赖绝对路径、不绑定盘符，也不要求用户设置 `PYTHONPATH`。本次只修改了移动脚本的运行说明和 import 启动逻辑，没有修改核心建模、训练、评估或 baseline 的业务逻辑。

## 4. 运行命令

移动后推荐运行命令：

```bash
python -m scripts.demo_random_env_test
python -m scripts.demo_task_model
python -m report_tools.export_task_timeline_sin
python -m legacy.nn_reuse_experiments.eval_nn_reuse
```

核心训练评估命令：

```bash
python train_ppo_ultimate.py --timesteps 1000
python eval_base.py --quick
```

## 5. 连通性测试结果

| 命令 | 是否成功 | 结果说明 |
|---|---|---|
| `python -m scripts.demo_random_env_test` | 否 | 当前 PowerShell 环境中 `python` 命令不可用，命令在进入项目代码前失败；不是代码整理导致的问题。 |
| `python -m scripts.demo_task_model` | 否 | 当前 PowerShell 环境中 `python` 命令不可用，命令在进入项目代码前失败；不是代码整理导致的问题。 |
| `python train_ppo_ultimate.py --timesteps 1000` | 否 | 当前 PowerShell 环境中 `python` 命令不可用，命令在进入项目代码前失败；不是代码整理导致的问题。 |
| `python eval_base.py --quick` | 否 | 当前 PowerShell 环境中 `python` 命令不可用，命令在进入项目代码前失败；不是代码整理导致的问题。 |
| bundled Python: `-m scripts.demo_random_env_test` | 否 | 已能按 `-m` 找到移动后的脚本和项目根目录模块，但运行时缺少外部依赖 `gymnasium`；不是移动导致的问题。 |
| bundled Python: `-m scripts.demo_task_model` | 是 | 任务模型 demo 成功运行，完成任务/功耗/PUE/成本等控制台输出。 |
| bundled Python: `train_ppo_ultimate.py --timesteps 1000` | 否 | 缺少外部依赖 `stable_baselines3`；不是代码整理导致的问题。 |
| bundled Python: `eval_base.py --quick` | 否 | 导入环境时缺少外部依赖 `gymnasium`；不是代码整理导致的问题。 |
| bundled Python: `py_compile` 四个移动脚本 | 是 | `scripts/demo_random_env_test.py`、`scripts/demo_task_model.py`、`report_tools/export_task_timeline_sin.py`、`legacy/nn_reuse_experiments/eval_nn_reuse.py` 语法检查通过。 |

## 6. 后续建议

- 新增 `grid_model/`，拆出 `ieee14_loader.py`、`opf_solver.py`、`emission_model.py`、`mef_calculator.py`、`lmp_calculator.py` 等电网真实性模块。
- 新增 `env_wrappers/`，用 wrapper 把 `IDCPriceEnv20D_ultimate.py` 的 IDC 购电需求接到 OPF/LMP/MEF，而不是继续膨胀环境大文件。
- 在 `config_ultimate.py` 中增加 `GRID_CONFIG`，集中管理 IEEE 14 节点、IDC 接入 bus、线路容量、机组成本、OPF 开关和 MEF/LMP reward 权重。
- 在 `eval_base.py` 中增加 LMP、MEF、电压越限、线路负载率、OPF 可行性、拥塞成本等指标。
- 保留 `train_ppo_ultimate.py` 作为单智能体 PPO baseline。
- 后续新增 `train_mappo.py` 和 MAPPO/CTDE 专用模块，不要直接覆盖 PPO 训练入口。
