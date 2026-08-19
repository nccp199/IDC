# 智算中心—微电网协同调度：MAPPO短训练准备度审计

> 审计日期：2026-07-28  
> 项目路径：`C:\Users\bulio\Desktop\IDC\ultimate_simplify`  
> HARL路径：`C:\Users\bulio\Desktop\IDC\HARL`  
> 审计方式：只读扫描；未运行测试或训练，未修改算法、环境或Bridge代码。

## 总结

环境、padding、有界动作和smoke链路均已具备；但仓库当前没有独立的MAPPO短训练入口。现有双Agent训练仅由pytest smoke手工驱动一次rollout和一次update。

若直接开始“3个随机种子 × 每个种子40次update”，主要缺口在训练编排、固定评估、分代checkpoint、完整日志和多次update诊断，而不在已经通过验证的环境、padding或有界动作实现。

扫描时项目仓库与HARL仓库均为clean状态。

## 一、仓库结构与调用链

### 1. 双Agent MAPPO核心目录与文件

| 职责 | 实际位置 |
|---|---|
| IDC物理环境 | `envs/idc_price_env.py` — `IDCPriceEnv20D` |
| 电网耦合、OPF、MEF | `env_wrappers/grid_coupled_env.py` — `GridCoupledEnv` |
| 基础环境构造 | `train/train_ppo_ultimate.py` — `make_single_env()` / `make_unmonitored_env()` |
| smoke环境构造入口 | `marl/tests/helpers.py` — `make_grid_env()`，固定使用实验场景`main` |
| 双Agent环境 | `marl/envs/idc_grid_multi_agent_env.py` — `IDCGridMultiAgentEnv` |
| IDC/BESS动作拼接与范围检查 | `marl/adapters/action_adapter.py` — `FlatActionAdapter` |
| HARL异构桥接 | `marl/bridges/harl_bridge.py` — `HarlIDCGridBridge` |
| HARL padding桥接 | `marl/bridges/harl_padded_bridge.py` — `HarlPaddedBridge` |
| BESS逐步虚拟动作诊断 | `marl/diagnostics/bess_virtual_action_monitor.py` |
| BESS PPO诊断 | `marl/diagnostics/bess_policy_diagnostics.py` |
| 当前唯一双Agent MAPPO入口 | `marl/tests/test_harl_mappo_smoke.py` |
| HARL runner | `C:\Users\bulio\Desktop\IDC\HARL\harl\runners\on_policy_ma_runner.py` — `OnPolicyMARunner` |
| HARL runner主循环 | `C:\Users\bulio\Desktop\IDC\HARL\harl\runners\on_policy_base_runner.py` |
| IDC/BESS actor | `harl/algorithms/actors/mappo.py`、`harl/models/policy_models/stochastic_policy.py` |
| centralized critic | `harl/algorithms/critics/v_critic.py`、`harl/models/value_function_models/v_net.py` |
| actor buffer | `harl/common/buffers/on_policy_actor_buffer.py` |
| EP critic buffer和GAE | `harl/common/buffers/on_policy_critic_buffer_ep.py` |
| HARL TensorBoard logger | `harl/common/base_logger.py` |
| checkpoint实现 | `harl/runners/on_policy_base_runner.py` — `save()` / `restore()` |
| HARL确定性评估 | `harl/runners/on_policy_base_runner.py` — `eval()` |
| HARL输出目录构造 | `harl/utils/configs_tools.py` — `init_dir()` |

项目中当前不存在：

- 独立的`train_mappo.py`或短训练脚本；
- 双Agent MAPPO正式配置文件；
- 项目侧MAPPO评估入口；
- 项目侧MAPPO runner子类或物理指标logger。

### 2. 当前真实调用链

```text
pytest
  → marl/tests/test_harl_mappo_smoke.py
  → run_harl_mappo_padding_smoke()
  → _smoke_algo_args()
  → env_factory()
     → make_grid_env()
     → make_unmonitored_env()
     → IDCPriceEnv20D
     → GridCoupledEnv
     → IDCGridMultiAgentEnv
     → HarlIDCGridBridge
     → HarlPaddedBridge
     → BESSVirtualActionMonitor
  → monkeypatch HARL make_train_env/get_num_agents
  → OnPolicyMARunner(...)
  → warmup()
  → 24 × collect()
     → MAPPO.get_actions()
     → StochasticPolicy
     → ACTLayer
     → BoundedDiagGaussian/AffineTanhNormal
  → ShareDummyVecEnv.step()
  → HarlPaddedBridge切出BESS真实第0维
  → IDCGridMultiAgentEnv.compose_action()
  → GridCoupledEnv.step()
  → IDCPriceEnv20D.step()
  → actor/critic buffer insert
  → compute_returns()/GAE
  → 两个actor依次update
  → centralized critic update
  → 单次BESS PPO诊断
  → save()
  → 新runner restore()
```

关键事实：smoke没有调用`runner.run()`，而是手工执行24步和一次`runner.train()`。

### 3. 入口和输出位置

- 环境构造入口：`marl/tests/test_harl_mappo_smoke.py::env_factory()`。
- HARL配置入口：`marl/tests/test_harl_mappo_smoke.py::_smoke_algo_args()`。
- MAPPO runner入口：`harl.runners.on_policy_ma_runner.OnPolicyMARunner`。
- actor实现：`harl.algorithms.actors.mappo.MAPPO`和`StochasticPolicy`。
- critic实现：`harl.algorithms.critics.v_critic.VCritic`和`VNet`。
- buffer：`OnPolicyActorBuffer`和当前EP状态对应的`OnPolicyCriticBufferEP`。
- 评估入口：HARL `OnPolicyBaseRunner.eval()`；项目当前未接入双Agent eval factory。
- BESS虚拟动作诊断：`BESSVirtualActionMonitor`和`compute_bess_policy_diagnostics()`。

HARL通用输出路径格式：

```text
<logger.log_dir>/
  gym/idc_bess_padding/mappo/<exp_name>/
    seed-<seed>-<timestamp>/
      config.json
      progress.txt
      logs/
        events.out.tfevents.*
        summary.json
      models/
        actor_agent0.pt
        actor_agent1.pt
        critic_agent.pt
```

已确认的smoke模型示例：

```text
C:\Users\bulio\Desktop\IDC\ultimate_simplify\runs\
idc_bess_mappo_padding_smoke\harl_results\gym\
idc_bess_padding\mappo\idc_bess_padding_smoke\
seed-07110-2026-07-28-10-57-46\models
```

BESS CSV单独位于：

```text
runs\idc_bess_mappo_padding_smoke\bess_virtual_action_diagnostics*.csv
```

`progress.txt`只写评估步数和平均reward，不是完整训练CSV。

## 二、当前MAPPO训练能力

| 能力 | 状态 | 依据与限制 |
|---|---|---|
| 连续多次policy update | **部分支持** | HARL `OnPolicyBaseRunner.run()`支持；当前项目入口只手工执行1次update |
| 指定总环境步数或update次数 | **部分支持** | `train.num_env_steps`可设置；没有直接`updates`参数，update数由整数除法计算 |
| 指定训练随机种子 | **已支持** | `seed.seed_specify`、`seed.seed`；HARL同步设置Python、NumPy、Torch种子 |
| 多rollout线程 | **部分支持** | HARL支持Dummy/Subproc；当前smoke工厂硬编码`ShareDummyVecEnv([env_factory])`，始终1线程 |
| 定期保存模型 | **部分支持** | `episode % eval_interval == 0`时保存；没有独立`save_interval`，且每次覆盖相同3个文件 |
| 保存optimizer状态 | **尚未支持** | `save()`只保存actor、critic和可选ValueNorm权重 |
| 从checkpoint恢复训练 | **部分支持** | 可载入网络权重；不能恢复optimizer、update计数、LR进度、RNG状态 |
| 重载并确定性评估 | **部分支持** | HARL `eval()`使用`deterministic=True`；smoke只验证BESS actor和critic权重，没有执行重载后评估 |
| TensorBoard日志 | **已支持** | HARL `BaseLogger`和BESS monitor均可写TensorBoard |
| CSV日志 | **部分支持** | BESS逐步诊断有CSV；训练损失和物理指标没有正式CSV |
| NaN/inf检查 | **部分支持** | smoke检查动作、训练info和诊断；环境检查reward；标准`runner.run()`没有统一逐update检查 |
| 动作范围检查 | **已支持** | 有界分布输出Box内动作；`FlatActionAdapter`再次硬校验`[0,1]`且不clip |

update换算为：

```text
updates = floor(num_env_steps / episode_length / n_rollout_threads)
```

因此40次update所需：

```text
num_env_steps = 40 × 24 × n_rollout_threads
```

首次使用1线程时为`960`。

## 三、训练指标记录情况

### 1. 训练指标

以下“完整记录”指标准HARL `runner.run()`路径。当前smoke没有调用`episode_log()`，所以其单次actor/critic训练标量未按通用logger完整落盘。

| 指标 | 状态 | 位置或说明 |
|---|---|---|
| episode reward | **已完整记录** | `BaseLogger.per_step()`累计，episode结束后写TensorBoard |
| IDC actor loss | **已完整记录** | `agent0/policy_loss` |
| BESS actor loss | **已完整记录** | `agent1/policy_loss` |
| critic loss | **已完整记录** | `critic/value_loss` |
| IDC entropy | **已完整记录** | `agent0/dist_entropy` |
| BESS entropy | **已完整记录** | `agent1/dist_entropy` |
| advantage | **已计算但未写入日志** | `OnPolicyMARunner.train()`中由returns减value_preds |
| return | **已计算但未写入日志** | `OnPolicyCriticBufferEP.returns` |
| actor gradient norm | **已完整记录** | `agent0/actor_grad_norm`、`agent1/actor_grad_norm` |
| critic gradient norm | **已完整记录** | `critic/critic_grad_norm` |
| PPO clip fraction | **当前不存在** | MAPPO只返回平均`ratio`，没有计算通用clip fraction |
| learning rate | **当前不存在** | optimizer中可读取，但logger未记录 |

两个Agent的HARL日志使用`agent0/`和`agent1/`前缀，不会相互覆盖；但标签没有直接写`idc`/`bess`语义。

### 2. BESS虚拟动作诊断

| 指标 | 状态 | 说明 |
|---|---|---|
| 真实第0维动作 | **已完整记录** | 逐步CSV及TensorBoard |
| 虚拟21维mean/std/norm | **已完整记录** | CSV包含mean、std、L2、abs mean/max；可选保存21个原始分量 |
| effective log_prob | **已完整记录** | 单次smoke update后写TensorBoard |
| virtual log_prob sum | **已完整记录** | 单次smoke update后写TensorBoard |
| full log_prob | **已完整记录** | 22维new log_prob求和 |
| effective ratio | **已完整记录** | 单次smoke update |
| full ratio | **已完整记录** | 22维per-dim ratio乘积 |
| ratio gap | **已完整记录** | 单次smoke update |
| clip disagreement | **已完整记录** | 单次smoke update |

限制：

- policy级指标只在smoke里显式调用一次；
- 标准多update `runner.run()`没有这个调用钩子；
- policy指标只写TensorBoard，没有写入诊断CSV；
- monitor捕获异常后只设置`last_error`，不会中断训练。

### 3. 物理指标

| 指标 | 状态 | 已有变量 |
|---|---|---|
| BESS SOC | **已完整记录** | BESS CSV：`bess_soc` |
| 充电功率 | **已完整记录** | BESS CSV，来源`bess_charge_power_kW` |
| 放电功率 | **已完整记录** | BESS CSV，来源`bess_discharge_power_kW` |
| 购电功率 | **可以从现有变量获取** | `info["P_grid_kW"]` / `grid_power_kW` |
| 能耗 | **可以从现有变量获取** | `grid_energy_kWh`、`idc_energy_kWh`及累计量 |
| 成本 | **可以从现有变量获取** | `cost`、`hourly_cost`、`total_cost` |
| 碳排放 | **可以从现有变量获取** | `carbon_emission`、`total_carbon_emission`、`grid_total_emission_kg` |
| 任务完成率 | **可以从现有变量获取** | `completion_rate`、`total_completed_work` |
| 最终积压 | **可以从现有变量获取** | 终止步`backlog_work`/`Q` |
| OPF成功率 | **可以从现有变量获取** | 每步`grid_opf_success`；尚未聚合成成功率 |
| MEF成功率 | **可以从现有变量获取** | 每步`grid_mef_success`；尚未聚合成成功率 |
| 电压越限 | **可以从现有变量获取** | `grid_voltage_violation_count/magnitude`、min/max voltage |
| 线路越限 | **可以从现有变量获取** | `grid_line_overload_count/magnitude`、max loading |

除SOC和BESS充放电功率外，上述物理量目前均未进入MAPPO TensorBoard或CSV logger。

## 四、当前实际MAPPO配置

当前真实入口没有加载HARL的`mappo.yaml`。它直接由`_smoke_algo_args()`构造完整字典，且运行产生的`config.json`与下列值一致。因此单Agent的`configs/config_ultimate.py:PPO_CONFIG`不参与MAPPO。

| 配置 | 实际值 |
|---|---:|
| `episode_length` | 24 |
| `num_env_steps` | 24 |
| rollout线程 | 1 |
| evaluation线程 | 1，但`use_eval=False` |
| actor PPO epoch | 1 |
| critic epoch | 1 |
| actor mini-batch数 | 1 |
| critic mini-batch数 | 1 |
| actor learning rate | 0.0005 |
| critic learning rate | 0.0005 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_param` | 0.2 |
| `entropy_coef` | 0.01 |
| `value_loss_coef` | 1 |
| `max_grad_norm` | 10.0 |
| actor/critic hidden layers | `[32, 32]` |
| 激活函数 | ReLU |
| 层归一化 | 输入LayerNorm；每层MLP后LayerNorm |
| recurrent | 关闭 |
| `recurrent_n` | 1，仅占位 |
| `data_chunk_length` | 24 |
| centralized state | EP，294维 |
| IDC HARL observation | 288维 |
| BESS HARL observation | padding后288维，真实164维 |
| IDC HARL action | 22维Box `[0,1]` |
| BESS HARL action | padding后22维Box `[0,1]`，真实有效维度1 |
| bounded Box开关 | `use_bounded_box_actions=True` |
| action aggregation | `prod` |
| actor参数共享 | `share_param=False` |
| actor更新顺序 | `fixed_order=True` |
| ValueNorm | 关闭 |
| LR decay | 关闭 |
| seed | `seed_specify=True`，默认7110 |
| 日志频率 | 每1个update |
| 配置中的评估/保存频率 | `eval_interval=1000` |
| 实际smoke保存 | 第1次update后手工`runner.save()` |
| 实际smoke评估 | 未执行 |

网络结构：

- IDC actor：`288 → 32 → 32 → 22维均值/方差 → affine-tanh → [0,1]`；
- BESS actor：同样是`288 → 32 → 32 → 22`；
- centralized critic：`294 → 32 → 32 → 1`；
- 两个actor独立，critic共享。

有界动作三环节一致：

1. 采样：`AffineTanhNormal.sample()`输出映射后的Box动作；
2. buffer：保存映射后的动作和包含Jacobian修正的逐维log probability；
3. `evaluate_actions()`：对buffer动作执行逆变换并以同一Jacobian公式重算log probability。

Bridge只在送入底层环境前把BESS动作切为第0维，不修改buffer中的完整22维动作。

## 五、短训练缺口分析

目标：

- 3个训练随机种子；
- 每个种子40次update；
- 每10次update保存一次；
- update 0/10/20/30/40固定场景评估；
- 记录训练、动作、BESS诊断和物理指标；
- final模型保存并重载评估。

### A. 无需修改，直接配置即可

- 单线程40次update：设置`episode_length=24`、`num_env_steps=960`；
- 每个训练seed分别构造runner；
- 两个独立actor和centralized critic更新；
- affine-tanh有界Box动作；
- 24步episode和1线程rollout对齐；
- TensorBoard基础loss、entropy、ratio、gradient norm；
- final actor/critic权重保存和权重加载；
- 确定性动作生成能力。

### B. 需要少量新增日志或配置

- 把smoke配置物化为正式短训练配置；
- 显式配置3个训练seed和固定评估seed；
- 记录advantage、return、learning rate和通用PPO clip fraction；
- 将物理`info`字段写入训练/评估CSV和TensorBoard；
- 给`agent0/agent1`增加`idc/bess`语义标签；
- 把BESS policy diagnostics同时写入CSV；
- 检查并上报`monitor.last_error`；
- 为每个seed、rollout thread和evaluation run建立不冲突的输出目录。

### C. 必须修改训练流程

- 新增正式MAPPO训练入口，不能继续以pytest作为训练程序；
- 建立项目自己的train/eval环境工厂，替代smoke monkeypatch；
- 在每次update后调用BESS policy diagnostics；
- 支持update 0评估；
- 支持update 10/20/30/40固定场景评估；
- checkpoint按update编号保存，否则现有`save()`会覆盖旧文件；
- final模型加载后执行真正的确定性评估；
- 若要求中断后精确续训，还需保存optimizer、update、RNG和LR状态。

### D. 潜在阻塞问题

- 当前smoke环境工厂始终只构造1个DummyVecEnv，修改`n_rollout_threads`不会自动增加环境；
- 当前没有双Agent eval factory，不能仅把`use_eval`改为true；
- 现有`eval_interval`同时控制评估和保存，且没有update 0；
- checkpoint文件名固定，会丢失10/20/30代模型；
- BESS policy diagnostics目前只覆盖一次update；
- 多线程monitor若直接沿用当前模式，会产生多个未标thread ID的CSV和各自局部`global_step`；
- OPF/MEF失败被转换为`success=False/message`后继续运行，而训练logger不记录这些字段；
- smoke常量记录上游SHA `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`，但当前实际HARL HEAD为`050ad6a294fe9f7572985dea910d59ea6d4f94b4`，即本地有界动作commit；训练元数据必须记录实际HEAD。

## 六、潜在风险

### 1. smoke配置误用

smoke的网络仅`[32,32]`、PPO epoch 1、单线程、无评估，适合管线验证，不应原样当作长期训练配置。短训练可暂时沿用其优化参数作为工程验证基线，但必须更换入口和调度。

### 2. 总环境步数与update次数

40次update不是固定960步，除非rollout线程为1。一般公式为：

```text
num_env_steps = 40 × 24 × n_rollout_threads
```

非整除部分会被runner静默舍弃。

### 3. rollout线程与24步episode

当前物理环境`horizon=24`，且smoke的`episode_length=24`，单线程对齐正确。多线程时每个update采集`24 × 线程数`条转换；需要确认每个worker使用独立seed并正确自动reset。

### 4. BESS虚拟动作诊断进入多次update

逐步虚拟动作CSV由Bridge自动记录；但log_prob、ratio、clip disagreement只在smoke手工update后计算一次，不会自动进入40次update循环。

### 5. full/effective log probability

当前计算位置正确：

- old log probability在update前从buffer复制；
- new log probability在update后对同一buffer动作重算；
- full log probability为22维之和；
- full ratio为逐维ratio乘积，与HARL `action_aggregation="prod"`一致；
- effective值只取第0维，是脱离训练图的反事实诊断，不替换正式MAPPO loss。

### 6. bounded Box动作三环节一致性

采样、buffer和`evaluate_actions()`使用同一Affine-Tanh分布及Jacobian修正，一致性已经实现。Bridge没有clip；动作adapter越界时直接报错。

### 7. 模型保存内容

当前保存两个actor和一个centralized critic，数量正确。但不含optimizer；周期保存覆盖旧文件。

### 8. 模型重载与动作空间

restore前会按当前环境动作空间和配置重新构造actor，因此必须保持：

```text
padded Box shape=(22,)
low=0
high=1
use_bounded_box_actions=True
```

restore不会自动读取旧run的配置；错误配置可能导致权重结构或分布buffer不匹配。

### 9. 两个Agent日志覆盖

`agent0/`、`agent1/`不会覆盖；BESS CSV也使用非覆盖命名。但多线程CSV缺少thread/worker标识，后续聚合容易混淆。

### 10. OPF/MEF异常

OPF异常被包装成失败结果，MEF也返回失败对象。并非完全静默，因为`info`包含success和message；但MAPPO logger当前忽略这些字段，因此训练运行层面会表现为“继续训练且没有告警”。

## 七、最终建议

### 1. 推荐的MAPPO短训练入口

新增独立入口：

```text
train/train_harl_mappo_short.py
```

由它构造train/eval环境、3个seed、40次update、诊断、checkpoint和final reload评估；不要复用pytest入口执行正式训练。

### 2. 推荐使用或复制的配置文件

新增：

```text
configs/harl_mappo_short.yaml
```

初始值复制`test_harl_mappo_smoke.py::_smoke_algo_args()`的实际配置，保留`use_bounded_box_actions=True`、EP state和24步episode。不要复制单Agent `PPO_CONFIG`，也不要直接使用缺少项目覆盖项的HARL通用`mappo.yaml`。

### 3. 建议新增或修改的文件清单

- 新增`train/train_harl_mappo_short.py`；
- 新增`configs/harl_mappo_short.yaml`；
- 新增`marl/runners/idc_mappo_runner.py`；
- 新增`marl/logging/idc_mappo_logger.py`；
- 新增`eval/eval_harl_mappo.py`；
- 新增短训练调度、checkpoint和日志相关测试；
- 仅在需要时小范围扩展BESS诊断的CSV输出。

### 4. 建议保持不动的文件清单

- `envs/idc_price_env.py`；
- `env_wrappers/grid_coupled_env.py`；
- `marl/envs/idc_grid_multi_agent_env.py`；
- `marl/adapters/action_adapter.py`；
- `marl/bridges/harl_bridge.py`；
- `marl/bridges/harl_padded_bridge.py`；
- HARL的`distributions.py`、`act.py`；
- HARL MAPPO loss、critic、GAE和buffer实现；
- `grid_model/opf_solver.py`和`grid_model/mef_calculator.py`。

### 5. 首次短训练建议命令

以下是新增正式入口后的推荐命令，当前仓库尚不能直接执行：

```powershell
cd C:\Users\bulio\Desktop\IDC\ultimate_simplify

$env:HARL_SOURCE_PATH="C:\Users\bulio\Desktop\IDC\HARL"
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'

& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  -m train.train_harl_mappo_short `
  --config configs/harl_mappo_short.yaml `
  --seeds 7110 7111 7112 `
  --updates 40 `
  --episode-length 24 `
  --rollout-threads 1 `
  --save-every-updates 10 `
  --eval-updates 0 10 20 30 40 `
  --eval-seed 2026
```

### 6. 开始短训练前必须完成的最小改动

- 建立正式训练入口和双Agent train/eval环境工厂；
- 将update数明确转换为`num_env_steps`；
- 实现update 0及10/20/30/40固定评估；
- checkpoint按seed和update分目录保存；
- 在每个update调用BESS policy diagnostics；
- 落盘训练、动作、物理和OPF/MEF指标；
- final模型重载后执行确定性评估；
- 启动前记录并校验实际HARL HEAD、动作空间、有界动作开关和诊断`last_error`。

## 附：关键版本信息

- 项目仓库扫描时HEAD：`e3e77c487a1e59ed6d891e5ae34fe4732c580754`。
- HARL当前HEAD：`050ad6a294fe9f7572985dea910d59ea6d4f94b4`。
- HARL当前commit标题：`feat: add bounded Box actions for on-policy actors`。
- smoke中记录的HARL上游基线SHA：`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`。
