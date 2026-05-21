# Module Reorganization Report

## 1. 本次移动的文件表

| 原路径 | 新路径 | 说明 |
| --- | --- | --- |
| `config_ultimate.py` | `configs/config_ultimate.py` | 全局配置 |
| `experiment_cases.py` | `configs/experiment_cases.py` | 实验 case 配置 |
| `data_loader.py` | `data_io/data_loader.py` | 外部数据读取 |
| `task.py` | `idc_model/task.py` | Task 数据结构 |
| `task_model.py` | `idc_model/task_model.py` | IDC 任务与能耗模型 |
| `power_model.py` | `idc_model/power_model.py` | IDC 功耗模型 |
| `IDCPriceEnv20D_ultimate.py` | `envs/idc_price_env.py` | Gymnasium 环境 |
| `ga_base.py` | `algorithms/baselines/ga_base.py` | GA baseline |
| `pso_base.py` | `algorithms/baselines/pso_base.py` | PSO baseline |
| `train_ppo_ultimate.py` | `train/train_ppo_ultimate.py` | PPO 训练入口 |
| `eval_base.py` | `eval/eval_base.py` | 统一评估入口 |

以下文件已在目标目录，保持现状：

| 路径 | 说明 |
| --- | --- |
| `scripts/demo_random_env_test.py` | 环境随机动作 demo |
| `scripts/demo_task_model.py` | 任务模型 demo |
| `report_tools/export_task_timeline_sin.py` | 报告导出工具 |
| `legacy/nn_reuse_experiments/eval_nn_reuse.py` | 历史神经网络复用实验 |

## 2. 新目录职责说明

| 目录 | 职责 |
| --- | --- |
| `configs/` | 全局配置、实验 case、后续场景配置 |
| `data_io/` | 电价、碳排、PUE/COP 等外部序列读取 |
| `idc_model/` | IDC 任务、功耗、能耗与调度基础模型 |
| `envs/` | 强化学习环境主体 |
| `algorithms/` | 算法总目录 |
| `algorithms/ppo/` | 后续 PPO 相关扩展 |
| `algorithms/baselines/` | GA、PSO 等 baseline |
| `train/` | 训练脚本入口 |
| `eval/` | 评估脚本入口 |
| `metrics/` | 后续 IDC、电网、碳排、任务指标 |
| `grid_model/` | 后续电网模型 |
| `env_wrappers/` | 后续环境包装与耦合层 |
| `scripts/` | demo、临时检查脚本 |
| `report_tools/` | 报告数据导出与可视化工具 |
| `legacy/` | 历史实验与兼容保留 |
| `legacy/nn_reuse_experiments/` | 历史 NN reuse 实验 |

所有新目录均已补充 `__init__.py`。

## 3. 兼容入口文件说明

根目录保留以下兼容入口，旧命令仍可使用：

| 根目录文件 | 转发目标 |
| --- | --- |
| `config_ultimate.py` | `configs.config_ultimate` |
| `data_loader.py` | `data_io.data_loader` |
| `task.py` | `idc_model.task.Task` |
| `task_model.py` | `idc_model.task_model.IDCEnergyTaskModel` |
| `power_model.py` | `idc_model.power_model.IDCPowerModel` |
| `IDCPriceEnv20D_ultimate.py` | `envs.idc_price_env.IDCPriceEnv20D` |
| `ga_base.py` | `algorithms.baselines.ga_base.main()` |
| `pso_base.py` | `algorithms.baselines.pso_base.main()` |
| `train_ppo_ultimate.py` | `train.train_ppo_ultimate.main()` |
| `eval_base.py` | `eval.eval_base.main()` |

这些文件只做 import 转发或 CLI main 转发，不包含业务逻辑。

## 4. import 修复说明

已将内部模块引用切换为包路径：

- `configs.config_ultimate`
- `configs.experiment_cases`
- `data_io.data_loader`
- `idc_model.task`
- `idc_model.power_model`
- `idc_model.task_model`
- `envs.idc_price_env`

已更新的主要文件包括：

- `configs/experiment_cases.py`
- `idc_model/task_model.py`
- `envs/idc_price_env.py`
- `algorithms/baselines/ga_base.py`
- `algorithms/baselines/pso_base.py`
- `train/train_ppo_ultimate.py`
- `eval/eval_base.py`
- `scripts/demo_random_env_test.py`
- `scripts/demo_task_model.py`
- `report_tools/export_task_timeline_sin.py`

## 5. 可用运行命令

新模块命令：

```bash
python -m train.train_ppo_ultimate
python -m eval.eval_base
python -m algorithms.baselines.ga_base --quick
python -m algorithms.baselines.pso_base --quick
python -m scripts.demo_random_env_test
python -m scripts.demo_task_model
```

旧兼容命令：

```bash
python train_ppo_ultimate.py
python eval_base.py
python ga_base.py --quick
python pso_base.py --quick
python IDCPriceEnv20D_ultimate.py
```

## 6. 检查结果

本机 PATH 中没有可直接调用的 `python` / `py` / `python3`，因此使用 Codex bundled Python 执行检查：

```bash
python -m py_compile <all .py files>
```

结果：通过，所有 `.py` 文件语法编译成功。

模块 import 检查结果：

| 模块 | 结果 |
| --- | --- |
| `configs.config_ultimate` | OK |
| `data_io.data_loader` | OK |
| `idc_model.task` | OK |
| `idc_model.power_model` | OK |
| `idc_model.task_model` | OK |
| `envs.idc_price_env` | FAIL: 缺少 `gymnasium` |
| `algorithms.baselines.ga_base` | FAIL: 缺少 `gymnasium` |
| `algorithms.baselines.pso_base` | FAIL: 缺少 `gymnasium` |
| `train.train_ppo_ultimate` | FAIL: 缺少 `stable_baselines3` |
| `eval.eval_base` | FAIL: 缺少 `gymnasium` |

以上失败均由当前 Python 环境缺少依赖导致，未修改业务逻辑规避依赖。

## 7. 后续建模升级建议

- `grid_model/` 放 IEEE14、OPF、LMP、MEF。
- `env_wrappers/` 放 `GridCoupledEnv`。
- `metrics/` 放 IDC、电网、碳排、任务指标。
- 后续新增 `algorithms/mappo/` 放 MAPPO+CTDE。
- 后续新增 `algorithms/attention/` 放注意力模块。
