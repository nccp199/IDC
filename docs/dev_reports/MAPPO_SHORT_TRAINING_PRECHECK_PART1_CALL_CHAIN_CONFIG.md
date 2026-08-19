# MAPPO短训练前检查——第一部分：训练调用链与配置来源

> 检查日期：2026-07-28  
> 项目路径：`C:\Users\bulio\Desktop\IDC\ultimate_simplify`  
> HARL路径：`C:\Users\bulio\Desktop\IDC\HARL`  
> 检查方式：只读扫描；未运行测试或训练，未修改环境、Bridge、padding、有界动作、MAPPO loss或GAE。

## 核心结论

当前仓库不存在不依赖pytest或测试模块的正式双Agent MAPPO训练入口。

HARL通用入口能够运行MAPPO，但不能直接创建本项目的双Agent环境。现有可运行链路位于`marl/tests/test_harl_mappo_smoke.py`，它通过测试内的运行时函数替换注入环境，并手工执行24步rollout和一次update，没有调用标准`runner.run()`。

第一部分判定：

> **3. 未通过，必须先建立正式训练入口。**

## A. 当前调用链

### A.1 所有可能的MAPPO入口

| 入口 | 文件、函数或类 | 启动命令 | 多次update | 性质 | 适合正式短训练 |
|---|---|---|---|---|---|
| pytest smoke | `marl/tests/test_harl_mappo_smoke.py`：`HarlMAPPOSmokeTest.test_one_stock_mappo_update_save_and_reload()` | `python -m pytest marl/tests/test_harl_mappo_smoke.py -q -s` | 否，固定1次 | 集成测试 | 否 |
| unittest执行同一测试 | 同上 | `python -m unittest marl.tests.test_harl_mappo_smoke.HarlMAPPOSmokeTest.test_one_stock_mappo_update_save_and_reload` | 否 | 同一测试的另一启动方式 | 否 |
| 直接调用smoke函数 | `marl/tests/test_harl_mappo_smoke.py::run_harl_mappo_padding_smoke()` | 只能通过自定义`python -c`或其他Python代码导入调用 | 否，函数内部固定1次 | 测试辅助API，无CLI | 否 |
| HARL官方入口 | `C:\Users\bulio\Desktop\IDC\HARL\examples\train.py::main()` | `python examples/train.py --algo mappo --env gym --exp_name ...` | 是，调用`runner.run()` | HARL通用训练入口 | 对本项目不适用 |
| 项目单Agent PPO入口 | `train/train_ppo_ultimate.py` | `python -m train.train_ppo_ultimate ...` | 是，但不是MAPPO | 单Agent Stable-Baselines3 PPO | 否 |
| shell/PowerShell脚本 | 无 | 无 | 无 | 仓库中没有MAPPO `.ps1/.sh/.bat/.cmd` | 否 |
| 项目正式MAPPO封装 | 无 | 无 | 无 | 尚未建立 | 否 |

HARL官方入口不适用于本项目的原因：

1. `--env gym`进入`harl/utils/envs_tools.py::make_train_env()`；
2. 该分支实例化`harl/envs/gym/gym_env.py::GYMEnv`；
3. `GYMEnv.__init__()`调用`gym.make(args["scenario"])`；
4. `GYMEnv`固定设置`n_agents=1`；
5. `idc_bess_padding`不是通过Gym注册的单Agent环境；
6. 因而即使加载smoke的`config.json`，没有环境替换仍不能创建IDC/BESS双Agent链路。

明确回答：

> **当前仓库不存在不依赖pytest的正式双Agent MAPPO训练入口。**

### A.2 当前真实smoke调用链

```text
[项目]
pytest / unittest
→ marl/tests/test_harl_mappo_smoke.py
  HarlMAPPOSmokeTest.test_one_stock_mappo_update_save_and_reload()

[项目]
→ run_harl_mappo_padding_smoke()
  读取HARL_SOURCE_PATH/HARL_RUNTIME_PATH/HARL_SMOKE_OUTPUT_DIR

[项目]
→ _smoke_algo_args(seed=7110, output_dir)
  直接构造完整algo_args字典

[项目]
→ env_factory()
  → marl/tests/helpers.py::make_grid_env(seed)
  → configs/experiment_cases.py::get_experiment_case("main")
  → train/train_ppo_ultimate.py::make_unmonitored_env()
  → make_single_env()
  → IDCPriceEnv20D(...)
  → GridCoupledEnv(...)

[项目]
→ IDCGridMultiAgentEnv(GridCoupledEnv)
→ HarlIDCGridBridge(IDCGridMultiAgentEnv)
→ HarlPaddedBridge(HarlIDCGridBridge)

[HARL]
→ ShareDummyVecEnv([env_factory])

[测试中的运行时替换]
→ patch harl.runners.on_policy_base_runner.make_train_env
→ patch harl.runners.on_policy_base_runner.get_num_agents

[HARL]
→ OnPolicyMARunner(args, algo_args, env_args)
  → OnPolicyBaseRunner.__init__()
  → 创建两个MAPPO actor
  → 创建两个OnPolicyActorBuffer
  → 创建一个VCritic
  → 创建一个OnPolicyCriticBufferEP

[smoke手工驱动，不调用runner.run()]
→ runner.warmup()
→ 24 × runner.collect(step)
→ runner.envs.step(actions)
→ runner.insert(data)
→ runner.compute()
→ runner.prep_training()
→ runner.train()

[HARL OnPolicyMARunner.train()]
→ advantages = returns - value_preds
→ actor_buffer[0] → IDC MAPPO.train()/update()
→ actor_buffer[1] → BESS MAPPO.train()/update()
→ centralized VCritic.train()/update()
```

### A.3 各层实际输入输出

| 层 | 文件、类、函数 | 关键输入 | 关键输出 | 所属 |
|---|---|---|---|---|
| 配置构造 | `test_harl_mappo_smoke.py::_smoke_algo_args()` | seed、output_dir | 完整`algo_args` | 项目 |
| 基础环境配置 | `marl/tests/helpers.py::make_grid_env()` | seed | `GridCoupledEnv` | 项目测试模块 |
| 物理环境实例化 | `train/train_ppo_ultimate.py::make_single_env()` | env/reward/data配置 | `IDCPriceEnv20D` | 项目 |
| 电网包装 | 同一函数中的`GridCoupledEnv(...)` | `IDCPriceEnv20D`、Grid配置 | Grid耦合环境 | 项目 |
| 双Agent包装 | `IDCGridMultiAgentEnv.__init__()` | `GridCoupledEnv` | IDC/BESS字典接口、centralized state | 项目 |
| HARL Bridge | `HarlIDCGridBridge` | 异构obs/action | 有序双Agent HARL接口 | 项目 |
| Padding | `HarlPaddedBridge` | IDC `(288,22)`、BESS `(164,1)` | 两Agent均为obs 288、action 22 | 项目 |
| VecEnv | `ShareDummyVecEnv.__init__()` | `[env_factory]` | 带线程维的数组接口 | HARL |
| Runner构造 | `OnPolicyBaseRunner.__init__()` | `args/algo_args/env_args` | actor、critic、buffers、envs | HARL |
| warmup | `OnPolicyBaseRunner.warmup()` | VecEnv reset | actor初始obs、critic初始state | HARL |
| rollout | `OnPolicyBaseRunner.collect()` | buffer中当前obs/state | values、actions、log_probs、RNN states | HARL |
| 环境step | `ShareDummyVecEnv.step_wait()` | `(threads,agents,action_dim)` | obs、state、reward、done、info | HARL/项目环境 |
| buffer写入 | `OnPolicyBaseRunner.insert()` | 一步transition | 两个actor buffer和EP critic buffer更新 | HARL |
| return/GAE | `OnPolicyBaseRunner.compute()` → `OnPolicyCriticBufferEP.compute_returns()` | 末状态value、rollout reward/mask | returns | HARL |
| advantage | `OnPolicyMARunner.train()` | returns、value_preds | advantages | HARL |
| IDC actor更新 | `self.actor[0].train()` | IDC actor buffer、advantages | IDC train info | HARL |
| BESS actor更新 | `self.actor[1].train()` | BESS actor buffer、advantages | BESS train info | HARL |
| critic更新 | `self.critic.train()` | EP critic buffer | critic train info | HARL |

实际更新顺序由`harl/runners/on_policy_ma_runner.py::OnPolicyMARunner.train()`中的：

```python
for agent_id in range(self.num_agents):
```

确定。结合`marl/specs/agent_specs.py`：

```python
AGENTS = ("idc", "bess")
```

实际顺序是：

```text
IDC actor → BESS actor → centralized critic
```

`fixed_order=True`虽然传入runner，但`OnPolicyMARunner.train()`没有读取该字段；MAPPO当前始终按`range(num_agents)`更新。

### A.4 环境创建入口

| 问题 | 结论 |
|---|---|
| `IDCPriceEnv20D`在哪里实例化 | `train/train_ppo_ultimate.py::make_single_env()`第85行 |
| `GridCoupledEnv`在哪里包装 | 同一函数第86–92行 |
| `IDCGridMultiAgentEnv`在哪里包装 | `test_harl_mappo_smoke.py::env_factory()`第131行 |
| `HarlIDCGridBridge`在哪里包装 | 同一行 |
| `HarlPaddedBridge`在哪里包装 | `env_factory()`第130–133行 |
| `ShareDummyVecEnv`在哪里创建 | 对`make_train_env`的lambda替换中，第138–140行 |
| 当前是否硬编码1个环境 | 是，`ShareDummyVecEnv([env_factory])`列表只有一个工厂 |
| 是否使用`n_rollout_threads`创建环境 | 否，patch lambda忽略所有位置参数和关键字参数 |
| 工厂能否由正式脚本直接复用 | 技术上可导入部分辅助函数，但不应；完整工厂嵌套在测试函数内，并依赖测试目录 |
| 是否依赖monkeypatch | 是；双Agent环境注入依赖替换HARL runner模块中的`make_train_env` |

`make_unmonitored_env()`适合被正式工厂复用，但`marl/tests/helpers.py::make_grid_env()`和smoke内部`env_factory()`不适合作为生产入口。

## B. 配置来源表

当前smoke不是“官方YAML → 项目配置 → 命令行覆盖”的多层配置体系。它直接构造完整Python字典并传给runner。

### B.1 实际覆盖顺序

算法配置：

```text
test方法读取环境变量
→ 调用run_harl_mappo_padding_smoke()
→ _smoke_algo_args()直接创建完整algo_args
→ args字面量和env_args字面量
→ copy.deepcopy(algo_args)
→ OnPolicyMARunner.__init__()
→ set_seed()/init_device()/init_dir()派生运行时状态
```

不存在：

```text
HARL mappo.yaml加载
→ 项目MAPPO配置合并
→ MAPPO命令行覆盖
```

物理环境配置是另一条链：

```text
configs/config_ultimate.py
  ENV_CONFIG / REWARD_CONFIG / DATA_CONFIG
→ get_experiment_case("main")深拷贝
→ make_single_env()合并IDC_SCALE_CONFIG和外部时间序列
→ 添加server_seed/task_seed
→ IDCPriceEnv20D
→ 另行传入GRID_CONFIG/GRID_REWARD_CONFIG/GRID_SCENARIO_CONFIG
```

| 配置类别 | 来源文件 | 函数/字段 | 是否实际生效 | 是否被覆盖 |
|---|---|---|---|---|
| MAPPO主配置 | `marl/tests/test_harl_mappo_smoke.py` | `_smoke_algo_args()` | 是，当前唯一完整MAPPO算法配置 | 没有后续超参数覆盖 |
| algorithm/env/experiment元数据 | 同文件第118行 | `args`字面量 | 是 | 否 |
| centralized state类型 | 同文件第119行 | `env_args["state_type"]="EP"` | 是 | 否；runner默认EP未被使用 |
| 环境scenario元数据 | 同文件第119行 | `idc_bess_padding` | 是，用于HARL路径命名 | 实际环境由patch工厂创建 |
| HARL官方MAPPO YAML | `C:\Users\bulio\Desktop\IDC\HARL\harl\configs\algos_cfgs\mappo.yaml` | HARL默认配置 | 否 | 未加载 |
| HARL Gym YAML | `C:\Users\bulio\Desktop\IDC\HARL\harl\configs\envs_cfgs\gym.yaml` | `scenario: Ant-v2` | 否 | 未加载 |
| 单Agent `PPO_CONFIG` | `configs/config_ultimate.py` | `PPO_CONFIG` | 否，不参与MAPPO | 不适用 |
| 物理环境配置 | `configs/config_ultimate.py` | `ENV_CONFIG/REWARD_CONFIG/DATA_CONFIG` | 是 | `main`场景当前不改值 |
| IDC规模配置 | 同文件 | `IDC_SCALE_CONFIG` | 是 | 在`make_single_env()`合并时优先于同名早期键 |
| Grid配置 | 同文件 | `GRID_CONFIG`等 | 是 | 否 |
| 命令行MAPPO超参数 | 无 | 无 | 否 | 不存在 |
| `HARL_SOURCE_PATH` | smoke测试第287行 | HARL源码路径 | 是 | 插入`sys.path[0]` |
| `HARL_RUNTIME_PATH` | smoke测试第290行 | 可选依赖路径 | 可选 | 插入`sys.path` |
| `HARL_SMOKE_OUTPUT_DIR` | smoke测试第291行 | smoke输出根目录 | 是；未设置时使用默认值 | 环境变量可覆盖默认目录 |
| `PYTHONPATH` | 进程环境 | 项目模块搜索路径 | 影响导入 | 不属于MAPPO配置 |
| `PYTHONDONTWRITEBYTECODE` | 进程环境 | 禁止`.pyc` | 不影响训练配置 | 无 |
| runner内部state默认 | `OnPolicyBaseRunner.__init__()` | `env_args.get("state_type","EP")` | 默认未触发，值已显式为EP | 无 |
| runner随机seed默认 | `set_seed()` | `seed_specify=False`时随机生成 | 未触发，当前为True | 无 |
| 训练环境替换 | smoke第135–145行 | patch `make_train_env` | 是，决定实际环境 | runner构造后恢复原函数 |
| Agent数量替换 | smoke第136–145行 | patch `get_num_agents` | 是，强制返回2 | runner构造后恢复原函数 |
| reload配置覆盖 | smoke第228–229行 | `reload_args["train"]["model_dir"]` | 只影响重载runner | 不影响首次训练update |

### B.2 `config.json`保存的内容

`harl/utils/configs_tools.py::save_config()`保存：

```python
{
    "main_args": args,
    "algo_args": algo_args,
    "env_args": env_args,
}
```

调用位置为`harl/runners/on_policy_base_runner.py::OnPolicyBaseRunner.__init__()`，在：

- `set_seed()`之后；
- `init_device()`之后；
- 运行目录建立之后；
- 环境创建之前。

因此：

- 它保存的是传入runner的最终`args/algo_args/env_args`；
- 当前没有YAML或CLI覆盖，所以基本等同于smoke完整字典；
- 若`seed_specify=False`，会保存`set_seed()`生成后的seed；
- 它不是HARL原始默认配置；
- 不保存monkeypatch；
- 不保存`HARL_SOURCE_PATH`或实际HARL HEAD；
- 不保存展开后的物理环境构造参数；
- 不保存`runner.run_dir/log_dir/save_dir`等动态派生路径；
- 首次runner的`train.model_dir`仍为`null`；
- reload runner会产生另一份配置，其中`model_dir`被设为首次runner的保存目录。

## C. 最终生效配置表

以下值由源码和已有成功smoke生成的`config.json`交叉确认。

| 配置键 | 最终值 | 来源 | 说明 |
|---|---:|---|---|
| `algorithm_name` / `args.algo` | `mappo` | `test_harl_mappo_smoke.py:118` | 未覆盖 |
| `env_name` / `args.env` | `gym` | 同上 | 仅作为HARL registry/logger键；实际不是`GYMEnv` |
| `experiment_name` | `idc_bess_padding_smoke` | 同上 | 未覆盖 |
| `env_args.scenario` | `idc_bess_padding` | 第119行 | 路径/任务名；实际环境由patch注入 |
| `episode_length` | 24 | `_smoke_algo_args().train` | 决定buffer长度；手工循环也硬编码24 |
| `num_env_steps` | 24 | 同上 | smoke不调用`runner.run()`，所以该值不控制循环 |
| `n_rollout_threads` | 1 | 同上 | VecEnv实际也硬编码为1 |
| `n_eval_rollout_threads` | 1 | `_smoke_algo_args().eval` | `use_eval=False`，未创建eval env |
| `use_eval` | `False` | 同上 | 未覆盖 |
| `eval_interval` | 1000 | `_smoke_algo_args().train` | 手工smoke不使用标准评估判断 |
| `log_interval` | 1 | 同上 | 手工smoke不调用标准`episode_log`分支 |
| `ppo_epoch` | 1 | `_smoke_algo_args().algo` | 每个actor一次epoch |
| `critic_epoch` | 1 | 同上 | centralized critic一次epoch |
| `actor_num_mini_batch` | 1 | 同上 | 未覆盖 |
| `critic_num_mini_batch` | 1 | 同上 | 未覆盖 |
| actor learning rate | 0.0005 | `_smoke_algo_args().model.lr` | 两个actor相同 |
| critic learning rate | 0.0005 | `_smoke_algo_args().model.critic_lr` | 未覆盖 |
| `gamma` | 0.99 | `_smoke_algo_args().algo` | 传入EP critic buffer |
| `gae_lambda` | 0.95 | 同上 | 传入EP critic buffer |
| `clip_param` | 0.2 | 同上 | actor PPO和value clip共同使用 |
| `entropy_coef` | 0.01 | 同上 | 未覆盖 |
| `value_loss_coef` | 1 | 同上 | 未覆盖 |
| `use_max_grad_norm` | `True` | 同上 | 未覆盖 |
| `max_grad_norm` | 10.0 | 同上 | actor/critic共同使用 |
| actor隐藏层 | `[32,32]` | `_smoke_algo_args().model.hidden_sizes` | 两个actor相同 |
| critic隐藏层 | `[32,32]` | 同上 | 与actor共享model配置，但不共享参数 |
| 激活函数 | ReLU | `activation_func="relu"` | `MLPLayer`每层Linear→ReLU→LayerNorm |
| feature normalization | `True` | smoke model配置 | 输入先经LayerNorm |
| 初始化 | `orthogonal_` | smoke model配置 | actor输出gain 0.01 |
| naive recurrent | `False` | smoke model配置 | 未启用 |
| recurrent policy | `False` | smoke model配置 | 未启用 |
| `recurrent_n` | 1 | smoke model配置 | RNN关闭，仅用于state数组维度 |
| `data_chunk_length` | 24 | smoke model配置 | RNN关闭时不参与feed-forward训练 |
| centralized state类型 | `EP` | `env_args["state_type"]` | 显式设置，不使用runner默认 |
| centralized state维度 | 294 | `HarlIDCGridBridge._validate_spaces()` | 两Agent共享同一294维state |
| IDC原始obs/action | 288 / 22 | `HarlIDCGridBridge` | Agent 0 |
| BESS原始obs/action | 164 / 1 | `HarlIDCGridBridge` | Agent 1 |
| IDC HARL侧obs/action | 288 / 22 | `HarlPaddedBridge` | 不需要补齐 |
| BESS HARL侧obs/action | 288 / 22 | `HarlPaddedBridge` | obs补124维零，动作仅第0维有效 |
| bounded Box开关 | `True` | `_smoke_algo_args().model.use_bounded_box_actions` | HARL默认YAML中没有该键 |
| action aggregation | `prod` | `_smoke_algo_args().algo` | 多维ratio按乘积聚合 |
| actor参数共享 | `False` | `share_param=False` | runner创建两个独立MAPPO实例 |
| 配置中的fixed order | `True` | `fixed_order=True` | MAPPO runner实际不读取该字段 |
| 实际actor更新顺序 | IDC → BESS | `OnPolicyMARunner.train()` | `range(2)`，随后更新critic |
| seed指定开关 | `True` | `_smoke_algo_args().seed` | 不触发随机seed替换 |
| seed | 7110 | `run_harl_mappo_padding_smoke(seed=7110)`默认参数 | 同时传入物理环境工厂 |
| CUDA | `False` | `_smoke_algo_args().device` | `init_device()`最终选择CPU |
| device | `cpu` | `init_device()`派生 | `cuda=False` |
| Torch线程 | 1 | smoke device配置 | `torch.set_num_threads(1)` |
| 日志根目录 | `<output_dir>/harl_results` | `_smoke_algo_args().logger` | 默认output为项目下`runs/idc_bess_mappo_padding_smoke` |
| 初次`train.model_dir` | `None` | smoke train配置 | 初始随机构造模型 |
| 模型输出目录 | `<log_root>/gym/idc_bess_padding/mappo/idc_bess_padding_smoke/seed-07110-<timestamp>/models` | `configs_tools.init_dir()` | 运行时派生，不写回`algo_args` |
| reload `train.model_dir` | 首次runner的`save_dir` | smoke第228–229行 | 只用于重载验证 |

当前实际HARL HEAD：

```text
050ad6a294fe9f7572985dea910d59ea6d4f94b4
feat: add bounded Box actions for on-policy actors
```

smoke常量记录的上游基线：

```text
b1af98b0dbab72a2eee9d160751cd09aedbb8ce2
```

后者不是当前实际执行HEAD。

## D. 问题清单

### D.1 阻塞正式短训练

1. **不存在正式双Agent MAPPO Python入口。**

   当前仅有`marl/tests/test_harl_mappo_smoke.py`。

2. **不存在生产级HARL train环境工厂。**

   当前实际工厂嵌套在smoke函数中，并从`marl/tests/helpers.py`导入环境辅助函数。

3. **双Agent环境注入依赖替换`make_train_env`。**

   不替换时，HARL的`env="gym"`会创建单Agent `GYMEnv`。

4. **当前入口不调用`runner.run()`。**

   `num_env_steps`、`log_interval`、`eval_interval`等标准runner控制项没有真正控制smoke主循环。

5. **没有正式配置解析入口。**

   MAPPO超参数全部硬编码在测试函数的Python字典中，没有正式YAML或CLI覆盖机制。

6. **没有正式eval环境工厂。**

   当前`use_eval=False`，且没有patch `make_eval_env`。直接改为`True`会进入HARL单AgentGym环境构造。

### D.2 不阻塞但必须修正

1. `HARL_COMMIT`记录的是上游基线，不是实际执行HEAD。
2. `env="gym"`只是借用HARL logger/registry名；正式配置应明确记录实际自定义环境类型。
3. `config.json`没有记录HARL源码路径、实际HEAD、运行时替换或最终环境维度。
4. `fixed_order=True`容易让人误以为它控制MAPPO顺序；实际顺序由`range(num_agents)`固定。
5. HARL官方`mappo.yaml`没有`use_bounded_box_actions`，正式入口若加载该YAML，必须显式补齐并验证此键。
6. `num_env_steps=24`在smoke里只是配置快照，不是执行次数的真实控制源。
7. `marl/tests/helpers.py::make_grid_env()`属于测试模块；正式代码不应反向依赖测试包。

### D.3 可以保持现状

- `train/train_ppo_ultimate.py::make_unmonitored_env()`和`make_single_env()`可作为正式双Agent工厂的底层环境构造函数；
- `IDCGridMultiAgentEnv`；
- `HarlIDCGridBridge`；
- `HarlPaddedBridge`；
- `ShareDummyVecEnv`作为首次单线程短训练VecEnv；
- HARL `OnPolicyMARunner`；
- HARL actor/critic buffer；
- HARL GAE和MAPPO actor/critic更新；
- 现有smoke继续作为一次update回归测试；
- 首次短训练继续使用`n_rollout_threads=1`。

### D.4 暂不处理

按本次范围不继续检查：

- 缓存和并行性能；
- reward设计；
- 完整日志系统；
- checkpoint格式和恢复策略；
- HAPPO、HGTA和Safe RL；
- 正式长训练参数调优。

## E. 第一部分通过判断

选择：

> **3. 未通过，必须先建立正式训练入口。**

原因不是MAPPO算法调用链不清楚。现有链路已经可以从环境构造一直追踪到两个actor和centralized critic更新。

未通过的原因是：

- 唯一可运行入口位于pytest测试；
- 只执行一次手工update；
- 环境创建依赖运行时函数替换；
- 正式HARL入口不能创建项目双Agent环境；
- 当前不存在稳定、可审计的配置加载和覆盖机制。

### E.1 update与环境步数换算

HARL标准公式位于`harl/runners/on_policy_base_runner.py::OnPolicyBaseRunner.run()`：

```python
episodes = (
    int(num_env_steps)
    // episode_length
    // n_rollout_threads
)
```

因此：

```text
updates =
floor(
    num_env_steps
    / episode_length
    / n_rollout_threads
)
```

非整除部分会被整数除法直接丢弃，runner不会为余数执行额外rollout。

| rollout线程 | episode长度 | 40 update所需num_env_steps |
|---:|---:|---:|
| 1 | 24 | 960 |
| 2 | 24 | 1920 |

当前smoke只执行一次update不是因为标准公式，而是因为它手工执行：

```python
runner.warmup()
runner.logger.init(1)
for step in range(24):
    runner.collect(step)
    runner.envs.step(actions)
    runner.insert(data)

runner.compute()
runner.prep_training()
runner.train()
```

`runner.train()`只调用一次，且完全没有调用`runner.run()`。

### E.2 现有smoke循环40次能否作为可信短训练

> **不能。**

理由：

1. 如果循环运行pytest 40次，每次都会重新创建runner并随机初始化模型，得到40个相互独立的一次update测试，而不是连续40次update。
2. 每次调用都会重新创建环境、buffer和输出目录。
3. `num_env_steps`并不控制该手工流程。
4. smoke主循环硬编码`range(24)`。
5. 当前流程在一次update后没有调用标准`after_update()`再进入下一rollout。
6. 环境仍由测试中的运行时函数替换注入。
7. pytest断言和保存/重载验证属于测试生命周期，不应承担训练编排。
8. 标准runner的update计算、episode循环和配置控制没有得到复用。

### E.3 smoke中的全部运行时替换

| patch对象 | 替换内容 | 原因 | 正式入口能否依赖 |
|---|---|---|---|
| `sys.modules["tensorboardX"]` | 用Torch `SummaryWriter`兼容类替换 | 避免依赖真实`tensorboardX`接口 | 不应；正式环境应使用明确依赖或项目适配器 |
| `sys.modules["setproctitle"]` | 空函数模块 | 避免smoke依赖进程标题包 | 不应 |
| `sys.modules["yaml"]` | 空模块 | smoke不加载YAML，但HARL模块导入需要`yaml`符号 | 不应；正式入口需要真实配置解析 |
| `harl.runners.on_policy_base_runner.make_train_env` | 返回`ShareDummyVecEnv([env_factory])` | 注入项目双Agent环境 | 不应；必须改为正式train factory或项目runner |
| `harl.runners.on_policy_base_runner.get_num_agents` | 固定返回2 | 强制双Agent数量 | 不应；正式VecEnv应可靠暴露`n_agents=2` |
| reload阶段同两项 | 再次替换train env和Agent数 | 重载runner也需要同一环境空间 | 不应 |

`get_num_agents`替换在当前`HarlPaddedBridge.n_agents=2`且`ShareDummyVecEnv`会转发该字段的情况下可能是冗余保护，但它确实参与当前smoke，正式入口应移除。

## F. 最小改动建议

只涉及训练入口和配置，不涉及本次排除的后续系统。

```text
新增：
- train/train_harl_mappo_short.py
  职责：
  - 独立Python CLI；
  - 加载正式MAPPO配置；
  - 解析seed、updates、rollout threads、输出目录覆盖；
  - 将updates换算为num_env_steps；
  - 循环3个训练seed；
  - 构造正式train/eval环境；
  - 调用OnPolicyMARunner标准训练循环或项目runner的等价标准循环；
  - 保存所有覆盖后的最终配置快照；
  - 记录实际HARL HEAD；
  - 启动前验证Agent数、obs/action/state维度。

- configs/harl_mappo_short.yaml
  职责：
  - 作为双Agent MAPPO唯一基础配置；
  - 从当前_smoke_algo_args()复制已验证参数；
  - 显式包含use_bounded_box_actions=True；
  - 区分updates和派生的num_env_steps；
  - 明确train/eval环境配置。

- marl/envs/harl_env_factory.py
  职责：
  - make_harl_train_env(seed, n_rollout_threads, env_config)；
  - make_harl_eval_env(seed, n_eval_rollout_threads, env_config)；
  - 构造IDCPriceEnv20D → GridCoupledEnv → IDCGridMultiAgentEnv
    → HarlIDCGridBridge → HarlPaddedBridge；
  - 返回正式ShareDummyVecEnv或ShareSubprocVecEnv；
  - 可靠暴露n_agents=2；
  - 不依赖marl/tests或运行时全局函数替换。

可选新增：
- marl/runners/idc_mappo_runner.py
  职责：
  - 在不修改HARL源码的情况下，将正式train/eval factory注入runner；
  - 消除对harl.runners.on_policy_base_runner模块全局函数的替换。

复用：
- train/train_ppo_ultimate.py::make_unmonitored_env()
- train/train_ppo_ultimate.py::make_single_env()
- marl/envs/idc_grid_multi_agent_env.py
- marl/bridges/harl_bridge.py
- marl/bridges/harl_padded_bridge.py
- HARL OnPolicyMARunner
- HARL OnPolicyActorBuffer
- HARL OnPolicyCriticBufferEP

保持不动：
- envs/idc_price_env.py
- env_wrappers/grid_coupled_env.py
- marl/adapters/action_adapter.py
- HARL MAPPO loss
- HARL GAE
- HARL bounded Box动作实现
- 现有test_harl_mappo_smoke.py；保留为回归测试，不作为训练入口
```

正式入口职责判断：

| 职责 | 是否需要 |
|---|---|
| 独立Python入口 | **必须** |
| 正式train环境工厂 | **必须** |
| 正式eval环境工厂 | **必须**，因为后续短训练计划包含固定评估 |
| YAML配置加载 | **强烈建议**；不是HARL算法硬要求，但应形成唯一可审计配置源 |
| 命令行覆盖 | **必须**，至少支持seed、updates、threads、output |
| update到`num_env_steps`转换 | **必须** |
| 多seed循环 | **必须**，对应3个训练seed目标 |
| 最终配置快照 | **必须**，且应在全部覆盖和派生后保存 |
| HARL实际HEAD记录 | **必须** |
| 启动前维度校验 | **必须**；确认2 Agent、obs 288、action 22、state 294 |
| 禁止pytest monkeypatch | **必须**；正式训练不能依赖测试模块或全局函数替换 |

## 关键文件索引

### 项目仓库

- `marl/tests/test_harl_mappo_smoke.py`
  - `_smoke_algo_args()`
  - `run_harl_mappo_padding_smoke()`
  - `HarlMAPPOSmokeTest.test_one_stock_mappo_update_save_and_reload()`
- `marl/tests/helpers.py`
  - `make_grid_env()`
- `train/train_ppo_ultimate.py`
  - `make_single_env()`
  - `make_unmonitored_env()`
- `configs/experiment_cases.py`
  - `get_experiment_case()`
- `configs/config_ultimate.py`
  - `ENV_CONFIG`
  - `DATA_CONFIG`
  - `IDC_SCALE_CONFIG`
  - `GRID_CONFIG`
  - `PPO_CONFIG`（不参与双Agent MAPPO）
- `marl/envs/idc_grid_multi_agent_env.py`
  - `IDCGridMultiAgentEnv`
- `marl/bridges/harl_bridge.py`
  - `HarlIDCGridBridge`
- `marl/bridges/harl_padded_bridge.py`
  - `HarlPaddedBridge`

### HARL仓库

- `examples/train.py`
  - `main()`
- `harl/utils/configs_tools.py`
  - `get_defaults_yaml_args()`
  - `update_args()`
  - `save_config()`
- `harl/utils/envs_tools.py`
  - `make_train_env()`
  - `make_eval_env()`
  - `set_seed()`
  - `get_num_agents()`
- `harl/envs/gym/gym_env.py`
  - `GYMEnv`
- `harl/envs/env_wrappers.py`
  - `ShareDummyVecEnv`
- `harl/runners/on_policy_base_runner.py`
  - `OnPolicyBaseRunner.__init__()`
  - `run()`
  - `warmup()`
  - `collect()`
  - `insert()`
  - `compute()`
- `harl/runners/on_policy_ma_runner.py`
  - `OnPolicyMARunner.train()`
- `harl/algorithms/actors/mappo.py`
  - `MAPPO.train()`
  - `MAPPO.update()`
- `harl/algorithms/critics/v_critic.py`
  - `VCritic.train()`
  - `VCritic.update()`
- `harl/common/buffers/on_policy_actor_buffer.py`
  - `OnPolicyActorBuffer`
- `harl/common/buffers/on_policy_critic_buffer_ep.py`
  - `OnPolicyCriticBufferEP.compute_returns()`
