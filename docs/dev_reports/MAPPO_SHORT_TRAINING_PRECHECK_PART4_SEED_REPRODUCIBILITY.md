# MAPPO短训练前检查——第四部分：随机种子与场景复现

## 0. 审计结论

本次检查只覆盖随机种子、场景复现、probe/train/eval隔离、算法初始化和正式1-update复现。没有修改seed设计、环境、物理模型、缓存、MAPPO、Bridge、checkpoint或任何训练代码；没有执行5-update或40-update训练。

最终判断选择：

> **1. 通过，单worker同seed可稳定复现。**

直接证据如下：

- 两个fresh正式环境使用`seed=7110`时，第一次reset后的raw observation、padded observation、centralized state、server参数、完整Task列表、外部曲线、BESS SOC、初始OPF/LMP/MEF和cache状态全部逐值一致。
- 两个fresh正式环境输入同一组24步合法动作后，observation、state、reward、done和列入检查的全部物理/电网/cache字段均逐值一致，最大差异均为`0.0`，且都只在第24步结束。
- 同一环境连续两次reset产生不同任务；重新创建同seed环境后，第一次reset恢复为原第一组任务。
- 两次同seed构造的IDC Actor、BESS Actor和centralized Critic初始参数SHA-256完全相同，optimizer初态完全相同。
- 两次独立进程的正式1-update中，所有已记录训练指标逐值相同，最终三个模型文件的SHA-256也完全相同。
- `seed=7111`会同时改变server参数、task场景、初始obs/state和三个网络的初始化hash。

当前没有阻塞单worker MAPPO短训练的seed问题。主要缺口是**复现实验元数据不充分**：`resolved_config.json`和`run_metadata.json`没有快照化实际物理/Grid/cache配置，特别是没有记录`cache_mef_load_bin_mw=0.01`；项目工作区当前有未提交/未跟踪实现，但metadata只记录Git HEAD，没有记录dirty状态或diff，因此仅凭现有metadata无法在另一checkout上完整还原本次运行。

---

## 1. 检查范围与证据

### 1.1 项目与版本

| 项目 | 实际值 |
| --- | --- |
| 项目路径 | `C:\Users\bulio\Desktop\IDC\ultimate_simplify` |
| 项目Git HEAD | `e3e77c487a1e59ed6d891e5ae34fe4732c580754` |
| HARL路径 | `C:\Users\bulio\Desktop\IDC\HARL` |
| HARL实际HEAD | `050ad6a294fe9f7572985dea910d59ea6d4f94b4` |
| HARL工作区 | clean |
| Python | `C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe` |
| device | CPU |
| PyTorch线程 | 1 |
| train seed | 7110 |
| eval seed | 2026（但当前`use_eval=false`） |

项目工作区不是clean；本次没有改动已有源文件，只新增本报告。现有dirty/untracked状态属于前序实现，包括正式训练入口、环境工厂、runner adapter、配置、缓存修正和测试。

### 1.2 源码证据范围

主要追踪文件：

- `train/train_harl_mappo_short.py`
  - `build_parser()`
  - `resolve_config()`
  - `split_runner_config()`
  - `prepare_training_environment()`
  - `run_training()`
  - `main()`
- `marl/envs/harl_env_factory.py`
  - `make_harl_single_env()`
  - `make_harl_train_env()`
  - `make_harl_eval_env()`
  - `_make_harl_vec_env()`
- `marl/runners/idc_mappo_runner.py`
  - `IDCOnPolicyMARunner.__init__()`
- `train/train_ppo_ultimate.py`
  - `_worker_seed()`
  - `make_single_env()`
  - `make_unmonitored_env()`
- `envs/idc_price_env.py`
  - `IDCPriceEnv20D.__init__()`
  - `IDCPriceEnv20D.reset()`
- `idc_model/power_model.py`
  - `IDCPowerModel.__init__()`
- `idc_model/task_model.py`
  - `_sample_task_parameters()`
  - `create_random_tasks()`
  - `create_demo_tasks()`
- `marl/envs/idc_grid_multi_agent_env.py`
  - `IDCGridMultiAgentEnv.reset()`
- `marl/bridges/harl_bridge.py`
  - `HarlIDCGridBridge.seed()`
  - `HarlIDCGridBridge.reset()`
- `marl/bridges/harl_padded_bridge.py`
  - `HarlPaddedBridge.seed()`
- `grid_model/grid_cache.py`
  - `GridResultCache`
- 固定HARL：
  - `harl/utils/envs_tools.py::set_seed()`
  - `harl/utils/models_tools.py::init_device()`
  - `harl/runners/on_policy_base_runner.py`
  - `harl/runners/on_policy_ma_runner.py`
  - `harl/common/buffers/on_policy_actor_buffer.py`
  - `harl/common/buffers/on_policy_critic_buffer_ep.py`
  - `harl/models/base/act.py`
  - `harl/models/base/distributions.py`

---

## A. seed调用链

### A.1 实际运行顺序

实际顺序和“先runner后环境”的概念图不同：正式入口先构造probe/train环境，之后才构造runner并设置算法全局随机源。

```text
CLI --seed 7110
  → train/train_harl_mappo_short.py::main()
  → resolve_config(..., seed=7110)
      → resolved["seed"]["seed"] = 7110
      → resolved["seed"]["seed_specify"] = true
  → run_training(resolved)
      │
      ├─ prepare_training_environment(resolved)
      │    → make_harl_train_env(seed=7110)       [probe]
      │    → make_harl_single_env(seed=7110)
      │    → make_unmonitored_env(..., seed=7110)
      │    → train_ppo_ultimate._worker_seed(7110, rank=0) = 7110
      │    → IDCPriceEnv20D(server_seed=7110, task_seed=7110)
      │         → IDCPowerModel.server_rng = default_rng(7110)
      │         → IDCPowerModel.task_rng   = default_rng(7110)
      │    → GridCoupledEnv → IDCGridMultiAgentEnv
      │      → HarlIDCGridBridge → HarlPaddedBridge
      │    → ShareDummyVecEnv([env_fn])
      │    → probe reset/validate/close
      │    → make_harl_train_env(seed=7110)       [fresh train]
      │
      ├─ 若use_eval=true：make_harl_eval_env(seed=eval.seed)
      │    当前配置eval.seed=2026；当前use_eval=false
      │
      ├─ split_runner_config(resolved)
      └─ IDCOnPolicyMARunner.__init__()
           → HARL set_seed({seed_specify:true, seed:7110})
               → random.seed(7110)
               → np.random.seed(7110)
               → os.environ["PYTHONHASHSEED"] = "7110"
               → torch.manual_seed(7110)
               → torch.cuda.manual_seed(7110)
               → torch.cuda.manual_seed_all(7110)
           → init_device(cpu)
               → torch.set_num_threads(1)
           → IDC Actor初始化
           → BESS Actor初始化
           → centralized Critic初始化
           → optimizer初始化
           → runner.run()
               → stochastic action sampling（PyTorch RNG）
               → buffer minibatch randperm（PyTorch RNG）
               → actor0 → actor1 → critic更新
```

### A.2 调用链表

| 随机对象 | 文件/函数 | seed来源 | 是否独立RNG | 是否可复现 |
| --- | --- | ---: | --- | --- |
| CLI seed | `train/train_harl_mappo_short.py::build_parser/main` | `--seed 7110` | 不适用 | 是 |
| resolved seed | `resolve_config()` | CLI覆盖YAML `seed.seed` | 单一配置值 | 是 |
| Python `random` | HARL `harl/utils/envs_tools.py::set_seed` | `algo_args.seed.seed` | 模块全局RNG | 是，当前路径未发现实际消费 |
| NumPy全局RNG | HARL `set_seed()` | 7110 | 模块全局RNG | 是 |
| PyTorch CPU RNG | HARL `set_seed()` | 7110 | PyTorch全局generator | 是，实测精确复现 |
| PyTorch CUDA RNG | HARL `set_seed()` | 7110 | CUDA全局generator | 已播种；本次CPU未验证 |
| server RNG | `idc_model/power_model.py::IDCPowerModel.__init__` | worker seed 7110 | `default_rng`独立实例 | 是 |
| task RNG | 同上 | worker seed 7110 | `default_rng`独立实例 | 是 |
| base Gym action/obs space | `train/train_ppo_ultimate.py::make_single_env` | worker seed 7110 | 每个space自身RNG | 是 |
| multi-agent/padded space | Bridge的`seed()`可播种，但正式工厂没有调用 | 当前未显式播种 | 每个space自身RNG | **未显式受控，但正式训练不调用space.sample()** |
| Actor初始化 | HARL模型初始化 | `torch.manual_seed(7110)`后的PyTorch RNG | 共享PyTorch全局RNG，按固定顺序消费 | 是 |
| Critic初始化 | `VCritic/VNet` | 同上 | 同上 | 是 |
| stochastic action | `ACTLayer`→bounded Box distribution `sample()` | PyTorch RNG | 同一全局RNG顺序推进 | 是 |
| minibatch排列 | HARL actor/critic buffer `torch.randperm` | PyTorch RNG | 同一全局RNG顺序推进 | 是 |
| pandapower OPF/MEF | `grid_model`与pandapower确定性求解 | 无随机seed | 不使用RNG | 当前CPU实测确定 |
| Grid cache | `GridResultCache` | 无随机seed | 每环境独立字典/LRU | 不引入随机性 |

### A.3 四个metadata seed的真实语义

| 字段 | 当前记录 | 实际含义 |
| --- | ---: | --- |
| `algorithm_seed` | 7110 | 传给HARL `set_seed()`，控制Python/NumPy/PyTorch全局随机源和网络初始化/动作采样/mini-batch |
| `environment_seed` | 7110 | 传给正式环境工厂的单一seed；主要通过worker seed继续传给server/task和底层space |
| `server_seed` | 7110 | 当前不是独立配置字段；metadata把同一个resolved seed换名记录，实际作为`default_rng(7110)`构造server RNG |
| `task_seed` | 7110 | 当前不是独立配置字段；metadata把同一个resolved seed换名记录，实际作为另一个`default_rng(7110)`构造task RNG |

四个字段当前数值确实都是7110，但它们不是四套CLI配置。server与task虽然seed数值相同，却是两个独立`numpy.random.Generator`对象，不共享状态。

两个`default_rng(7110)`初始bit generator状态相同；如果执行完全相同的采样调用，会产生相同数列。但server与task调用的分布、shape和顺序不同：server依次采样idle、max和capacity向量；task依次采样类型、arrival、duration、load profile、deadline和priority。因此两者语义结果不同，随后RNG状态也不同。

`IDCPriceEnv20D.reset(seed=...)`调用Gym基类reset，但**不会重建**`model.server_rng`或`model.task_rng`。任务生成继续推进构造时创建的`task_rng`，这正是连续episode场景变化且整个同seed运行可复现的原因。

---

## B. 随机源清单

| 随机源 | seed来源 | 是否独立 | 是否受控 |
| --- | ---: | --- | --- |
| Python `random` | 7110 | 全局 | 已受控；当前正式路径未发现消费点 |
| NumPy legacy全局RNG | 7110 | 全局 | 已受控；正式环境的server/task不依赖它 |
| PyTorch CPU RNG | 7110 | 全局 | 已受控；两次进程实测精确一致 |
| PyTorch CUDA RNG | 7110 | CUDA全局 | 已播种，本次未使用/未验证 |
| server `Generator` | 7110 | 是 | 已受控 |
| task `Generator` | 7110 | 是 | 已受控 |
| base action space RNG | 7110 | 是 | 已受控 |
| multi/padded action space RNG | 未显式调用Bridge.seed | 是 | 当前训练无影响；未来若调用`space.sample()`有风险 |
| bounded Gaussian sample | PyTorch RNG | 否，使用PyTorch全局RNG | 已受控 |
| actor/critic初始化 | PyTorch RNG | 否，固定构造顺序 | 已受控 |
| feed-forward mini-batch `torch.randperm` | PyTorch RNG | 否 | 已受控 |
| Grid cache | 无随机源 | 每环境独立 | 确定性 |
| OPF/MEF/pandapower | 无显式随机源 | 不适用 | 当前输入和CPU运行下确定性 |
| `eval/eval_base.py::action_random(rng=None)` | 未传时会`default_rng()` | 独立但熵播种 | **正式MAPPO入口未调用**；其他评估helper未来风险 |

HARL的`share_param=false`且项目runner固定actor更新顺序为IDC后BESS。`OnPolicyMARunner`中`torch.randperm(num_agents)`只位于share-param分支，当前不执行。`fixed_order=true`也与当前固定顺序一致。

---

## C. 场景组成表

| 场景组成 | 当前是否随机 | 由什么控制 |
| --- | --- | --- |
| server idle/max功率 | 是 | `server_rng=default_rng(worker_seed)` |
| server capacity/总算力 | 是 | 同一server RNG |
| task arrival/type/workload | 是 | `task_rng=default_rng(worker_seed)` |
| task duration/deadline/priority | 是 | 同一task RNG |
| task interruptible/parallelizable | 类型模板固定；随随机类型间接变化 | 固定task profile + task RNG选择类型 |
| task load profile | 是 | task RNG uniform采样 |
| price | 否 | 当前DATA_CONFIG未指定CSV，使用代码内固定24小时分时曲线 |
| carbon | 否 | 当前DATA_CONFIG未指定CSV，使用代码内固定24小时曲线 |
| temperature | 否 | 代码内确定性正弦曲线 |
| PV | 否 | DATA_CONFIG启用固定解析生成的日照正弦曲线 |
| dynamic grid load scale | 否 | 固定CSV `data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale.csv` |
| BESS初始SOC | 否 | `ENV_CONFIG.bess_soc_init=0.50`，reset时固定恢复 |
| IEEE14网络 | 否 | `pandapower.networks.case14()`及确定性电压限值协调 |
| OPF/LMP/MEF | 否 | 当前网络、负荷、动作和solver输入的确定性函数 |
| cache命中序列 | 否 | fresh环境、确定性key和确定性调用顺序；每环境独立 |
| IDC Actor初始参数 | 是 | PyTorch RNG / algorithm seed |
| BESS Actor初始参数 | 是 | 同上，按固定第二个actor构造顺序消费 |
| Critic初始参数 | 是 | 同上，按固定构造顺序消费 |
| stochastic actions | 是 | bounded Box分布 + PyTorch RNG |
| deterministic actions | 否（给定模型和obs） | 分布mode |

不同seed测试确认：server、task、初始obs和state会变化；price、carbon、temperature、PV、dynamic grid scale和BESS初始SOC保持相同。这是当前场景设计的真实语义，不是seed失效。

---

## D. 相同seed复现结果

### D.1 测试A：相同seed初始环境

两套全新正式`ShareDummyVecEnv`：

```text
env_A(seed=7110)
env_B(seed=7110)
```

| 比较项 | 结果 |
| --- | --- |
| raw observation | bitwise一致 |
| true IDC/BESS observations | bitwise一致 |
| padded observations | bitwise一致 |
| centralized state | bitwise一致 |
| server P_idle/P_max/C_server | 逐值一致 |
| 总算力/总idle/总max功率 | 逐值一致 |
| Task数量及完整字段 | 逐项一致 |
| price/carbon/temperature/PV | 逐值一致 |
| dynamic grid scale | 逐值一致 |
| BESS SOC | 一致 |
| cache初始统计 | 一致 |
| 初始OPF/MEF成功 | 都为true |

初始网格代表值：

| 指标 | 值 |
| --- | ---: |
| LMP | 39.85129916715733 |
| MEF plus | 319.0601987132686 |
| MEF minus | 0.0 |
| min voltage | 1.0150095659503928 |
| max voltage | 1.0816054669074937 |
| max line loading | 1.2184103705145082 |
| network loss MW | 9.059843920881548 |

### D.2 测试B：相同seed、相同24步动作

两套fresh正式VecEnv使用完全相同的确定性合法动作序列；cache均开启且不共享。

| 比较项 | 结果 |
| --- | --- |
| 24步obs | bitwise一致 |
| 24步state | bitwise一致 |
| 24步reward | bitwise一致 |
| done | 一致；只在第24步为true |
| backlog/Q | 最大差异0.0 |
| task completion | 最大差异0.0 |
| SOC/充放电功率 | 最大差异0.0 |
| IDC/grid功率 | 最大差异0.0 |
| cost/carbon | 最大差异0.0 |
| LMP/MEF plus/minus | 最大差异0.0 |
| voltage/line loading/loss | 最大差异0.0 |
| OPF/MEF成功标志 | 逐步一致 |
| OPF/MEF cache hit | 逐步一致 |
| 24步reward sum | 两者均为-30.0833902657032 |

HARL `ShareDummyVecEnv.step_wait()`在第24步done后会自动reset并把终止obs/state写入info；两套环境的这一自动reset行为也一致。

### D.3 连续reset语义

| 检查 | 结果 |
| --- | --- |
| 同一环境reset 1与reset 2的任务是否不同 | 是 |
| task RNG是否正常向前推进 | 是 |
| fresh同seed环境第一次reset是否恢复原任务 | 是 |
| fresh同seed第一次obs是否恢复 | 是 |
| 两个fresh环境task RNG是否为不同对象 | 是 |
| 两个fresh环境server RNG是否为不同对象 | 是 |

当前行为适合训练：同一次运行中各episode任务场景变化；重新启动相同seed实验时，整个任务序列恢复一致。

### D.4 算法初始化复现

两次使用同一resolved配置、各自fresh环境和fresh runner构造，不执行训练：

| 模块 | seed=7110初始参数SHA-256 | 两次是否相同 |
| --- | --- | --- |
| IDC Actor | `f43af350d9c2d12664cd07267a6edb694714846ea5a14b78620d9daf7c6df685` | 是 |
| BESS Actor | `d790e1a2da22fc6f1a440d5fb9fb02262a42487b438b46a2357896d551aaaabc` | 是 |
| centralized Critic | `c642f3f01677a80fb117112bc88e7aad6e24fbf3d65c0d218badf4aa1e2100f8` | 是 |

optimizer初始state均为空，actor0/actor1/critic的参数组、学习率、eps、weight decay及参数数量逐项一致。

相同obs与相同重播种后的动作结果：

- 两个runner的deterministic action逐值一致，最大差异0.0。
- 两个runner的stochastic action和log_prob逐值一致，最大差异0.0。
- 同一模型不重播种连续采样两次得到不同动作；这是正确的RNG推进，不是不可复现。

### D.5 两次正式1-update复现

两次运行均为独立Python进程、独立probe/train/runner/cache、独立输出目录，CPU、线程数、配置和seed相同，没有读取checkpoint。

| 指标 | run_A | run_B | 结论 |
| --- | ---: | ---: | --- |
| average step reward | -1.1291409730911255 | -1.1291409730911255 | 完全一致 |
| episode reward | -27.099385746754706 | -27.099385746754706 | 完全一致 |
| IDC policy loss | -4.967053879312289e-09 | -4.967053879312289e-09 | 完全一致 |
| IDC entropy | -8.970846176147461 | -8.970846176147461 | 完全一致 |
| IDC grad norm | 13.609609603881836 | 13.609609603881836 | 完全一致 |
| IDC ratio | 1.0 | 1.0 | 完全一致 |
| BESS policy loss | 7.450580596923828e-08 | 7.450580596923828e-08 | 完全一致 |
| BESS entropy | -8.96863079071045 | -8.96863079071045 | 完全一致 |
| BESS grad norm | 15.61442756652832 | 15.61442756652832 | 完全一致 |
| BESS ratio | 0.9999999403953552 | 0.9999999403953552 | 完全一致 |
| critic value loss | 82.437255859375 | 82.437255859375 | 完全一致 |
| critic grad norm | 139.9957275390625 | 139.9957275390625 | 完全一致 |

最终模型文件：

| 文件 | 两次共同SHA-256 | 是否完全相同 |
| --- | --- | --- |
| `actor_agent0.pt` | `433C5F81F96143E66810F6B26828C6DE89C45BD916A427BA6ED1E740BB6985F5` | 是 |
| `actor_agent1.pt` | `6E1709ABF7EF39E56386F34F892DF776B15F541AABBEC5D4EE49FE59DA177E2E` | 是 |
| `critic_agent.pt` | `21D242998343E962905192159532D78EE3B887820FCE8631A43E8979BD4A18FE` | 是 |

正式logger没有持久化每步reward或首个episode Task明细，因此这两项不能从两个run的输出文件逐行重放比较；但独立环境测试已经逐步验证同seed+同动作的reward与Task一致，且正式训练的全部可见指标和最终模型文件均字节级一致。这足以把本次正式1-update判定为“完全一致”。

---

## E. 不同seed差异结果

### E.1 环境seed 7110与7111

| 内容 | 是否变化 |
| --- | --- |
| server参数 | 是 |
| C_server最大绝对差异 | 1198.0654321110096 |
| Task序列与参数 | 是 |
| 初始observation | 是 |
| 初始centralized state | 是 |
| price | 否 |
| carbon | 否 |
| temperature | 否 |
| PV | 否 |
| dynamic grid scale | 否 |
| BESS初始SOC | 否 |

### E.2 模型seed 7110与7111

seed=7111只构造runner，不执行第三次update：

| 模块 | seed=7111初始SHA-256 | 与7110是否不同 |
| --- | --- | --- |
| IDC Actor | `481570c4c975c2106dbdd37e212982c18934c8c03cda646328110752348f5e9d` | 是 |
| BESS Actor | `2fd83e0320c7d792d8add4e1417c2fc11248dd2cb6fdb98a7e3eeb28a84361f0` | 是 |
| centralized Critic | `26231e0cc9194d81730072c32d3d1f290393d31a39c9f0a2d68816ee913cab8b` | 是 |

因此不同seed既改变随机环境场景，也改变算法初始参数。本次按要求没有运行seed=7111的第三次update。

---

## F. probe/train/eval隔离

| 检查 | 结果 |
| --- | --- |
| probe与train是否同一个环境对象 | 否 |
| probe与train是否共享task RNG | 否 |
| probe reset是否消耗train首个场景 | 否，实测train首个场景等于独立fresh环境首个场景 |
| train与eval是否共享环境或task RNG | 否 |
| 默认train seed | 7110 |
| 默认eval seed | 2026 |
| 当前是否实际创建eval | 否，`use_eval=false` |
| 默认eval场景是否与train首场景不同 | 是 |
| 如果人为令eval seed=7110，首场景是否与train相同 | 是 |

`prepare_training_environment()`先创建probe、reset并关闭，再用相同参数重新构造fresh train环境。每次构造都会生成独立server/task RNG和独立Grid cache，因此probe不会推进train状态。

`make_harl_eval_env()`也新建独立对象。当前YAML明确设置`eval.seed=2026`，所以不存在默认`train seed = eval seed`问题。若未来手工将两者设为相同seed，两边虽然不共享RNG对象，但会从相同状态开始并生成相同场景序列；这会使评估场景与训练场景重复。固定评估建议继续使用独立且明确记录的eval seed或固定场景清单。

---

## G. 未受控随机源与风险

### G.1 已受控

- server和task均使用显式seed的独立`default_rng`。
- HARL统一设置Python、NumPy、PyTorch CPU和CUDA seed。
- 当前CPU配置设置`torch_threads=1`。
- actor、critic、bounded stochastic action和mini-batch排列均由已播种的PyTorch RNG控制。
- probe/train/eval各自拥有独立环境、RNG和cache。
- Grid cache、Grid key、OPF/MEF调用链没有随机采样。

### G.2 当前无影响但未来有风险

1. **wrapper action/observation spaces没有被正式工厂显式seed。**
   - `make_single_env()`播种的是底层Grid/base space。
   - `IDCGridMultiAgentEnv`和`HarlPaddedBridge`之后创建了新space。
   - Bridge已经实现`seed()`，但`marl/envs/harl_env_factory.py::_make_harl_vec_env()`没有调用它。
   - 当前MAPPO动作来自PyTorch Actor，不调用`space.sample()`，因此两次1-update仍精确一致。

2. **`PYTHONHASHSEED`在解释器启动后才写入环境变量。**
   - HARL `set_seed()`中的赋值不能回溯改变当前Python进程已经确定的hash salt。
   - 当前调用链未发现依赖set/dict hash随机顺序的训练逻辑，两个独立进程也精确一致。
   - 若未来引入基于set遍历、动态插件注册或多进程，应该在启动Python前设置并记录该变量。

3. **CUDA确定性未被本次验证。**
   - HARL在CUDA模式下设置cuDNN benchmark=false和deterministic=true，也播种全部CUDA RNG。
   - 没有启用`torch.use_deterministic_algorithms(True)`；跨GPU/驱动/算子仍可能出现非确定性。
   - 当前正式短训练配置为CPU，不影响本结论。

4. **`eval/eval_base.py::action_random()`在没有显式rng时调用无seed的`default_rng()`。**
   - 当前正式MAPPO入口与HARL deterministic eval不调用该helper。
   - 未来若把该helper接入评估，必须显式传入RNG。

### G.3 明显未受控且影响当前正式路径

未发现。

---

## H. metadata充分性

### H.1 已记录

实际`run_metadata.json`已包含：

- `algorithm_seed=7110`
- `environment_seed=7110`
- `server_seed=7110`
- `task_seed=7110`
- `harl_scenario=idc_bess_padding`
- `experiment_case=main`
- `grid_scenario_source`及加载消息
- 项目Git HEAD
- HARL实际/期望HEAD及upstream commit
- Python executable/version
- PyTorch version
- device
- updates、num_env_steps、episode_length、rollout threads
- agent顺序、obs/action/state维度
- bounded Box与action aggregation标志
- 输出目录、模型文件、完成状态

实际`resolved_config.json`保存了CLI覆盖后的完整HARL短训练YAML，并追加runtime段，因此algorithm seed、eval seed、训练超参数和设备配置可追踪。

### H.2 缺失或语义不足

| 缺口 | 影响 |
| --- | --- |
| 没有实际`ENV_CONFIG/REWARD_CONFIG/DATA_CONFIG/IDC_SCALE_CONFIG`快照 | `experiment_case=main`不足以恢复当时的真实物理参数 |
| 没有`GRID_CONFIG/GRID_REWARD_CONFIG/GRID_SCENARIO_CONFIG`快照 | Grid场景只记录来源路径，未记录完整解析配置 |
| 没有`GRID_CACHE_CONFIG`快照 | 无法从run文件确认OPF/MEF分箱、scope、clear策略 |
| 未记录`cache_mef_load_bin_mw=0.01` | 本次第三部分新规则没有进入resolved config或metadata |
| 未记录Grid CSV内容hash | 相同路径内容变更后无法察觉 |
| 项目只记录Git HEAD，不记录dirty状态/diff | 当前正式入口和若干实现为untracked/modified，仅凭HEAD无法恢复实际代码 |
| 未记录NumPy、Gymnasium、pandapower及solver版本 | 跨机器复现OPF数值时证据不足 |
| 四个seed字段看起来独立，实为同一配置值的四个别名 | metadata没有说明server/task是独立RNG实例 |
| 未记录进程启动时真实`PYTHONHASHSEED` | HARL运行时写入值不等于解释器实际hash seed |
| 未持久化每步reward/Task场景标识 | 无法仅从run artifacts逐步审计rollout场景 |

结论：现有metadata足以解释当前HARL配置和主要seed行为，但**不足以独立完整复现实验**。最严重的不是随机源，而是物理/Grid/cache配置和dirty源码状态没有快照。

---

## I. 问题分类

### I.1 阻塞MAPPO短训练

无。当前CPU、单worker、fresh输出目录条件下，同seed环境、模型初始化和正式1-update均精确复现。

### I.2 必须在正式长训练前修正

1. `run_metadata.json`/`resolved_config.json`必须记录实际解析后的物理、Grid和cache配置，包括`cache_mef_load_bin_mw=0.01`。
2. 必须记录项目工作区是否dirty；正式可发布实验应对应可恢复的commit，或至少保存diff/源码快照。当前只记录HEAD会遗漏本次实际运行的大量代码。
3. 应记录关键数值依赖版本和Grid CSV内容hash，避免相同seed但不同solver/数据得到不同结果。

### I.3 建议改进

1. 正式工厂显式调用最终Bridge的`seed()`，使multi-agent和padded spaces的RNG语义与metadata一致；这不改变当前Actor采样。
2. 在Python启动前设置并记录`PYTHONHASHSEED`，不要只在HARL `set_seed()`中运行时赋值。
3. 增加固定的seed复现回归测试，覆盖fresh env、连续reset、初始化hash和两次1-update关键指标。
4. 若未来启用GPU，增加CUDA设备上的确定性验证及算子限制。
5. 若未来启用eval，保持独立eval seed，并记录固定评估场景清单/场景hash。

### I.4 可以保持现状

- 单一CLI seed同时作为algorithm/environment/server/task seed；本阶段不必拆分四套CLI。
- server/task使用相同数值但不同`default_rng`实例。
- 同一环境reset时推进task RNG，而不是每次重播种。
- probe关闭后重建fresh train环境。
- 默认`train seed=7110`、`eval seed=2026`。
- CPU和`torch_threads=1`。
- cache按环境独立，且跨episode保留；其命中序列在fresh同seed运行中可复现。
- 当前固定price/carbon/temperature/PV/grid scale/BESS初始SOC。

---

## J. 第四部分判断

选择：

> **1. 通过，单worker同seed可稳定复现。**

判断依据：

1. 环境初始状态与24步物理轨迹均零差异。
2. 连续reset正确推进，fresh同seed正确恢复。
3. probe/train/eval对象和RNG隔离正确。
4. Actor/Critic初始化和随机动作在重播种条件下精确复现。
5. 两个独立正式1-update进程的全部已记录指标和最终模型文件完全一致。
6. 不同seed确实改变环境随机场景和网络初始化。

这个“通过”只针对当前已验证的CPU、单worker正式MAPPO路径。metadata充分性问题不否定运行时确定性，但必须在正式长训练/论文结果前补齐。

---

## K. 最小修正建议（本次未实施）

```text
建议修改：
- train/train_harl_mappo_short.py
  职责：把实际ENV/REWARD/DATA/IDC scale/Grid/Grid reward/Grid scenario/
        Grid cache配置写入resolved_config和run_metadata；记录项目dirty状态、
        关键依赖版本、数据文件hash及进程启动时PYTHONHASHSEED。

- marl/envs/harl_env_factory.py
  职责：可选地对最终HarlPaddedBridge显式调用seed(seed)，使所有新建Gym
        spaces都具有明确seed；保持server/task RNG设计不变。

- configs/harl_mappo_short.yaml
  职责：仅补充seed语义注释，明确train seed、固定eval seed以及四个metadata
        名称当前来自同一train seed；不拆分四套CLI参数。

建议新增：
- marl/tests/test_harl_seed_reproducibility.py
  职责：固定验证同seed fresh reset、24步轨迹、连续reset、probe/train/eval
        隔离、Actor/Critic初始化hash及正式1-update关键指标。

保持不动：
- envs/idc_price_env.py
- idc_model/power_model.py
- idc_model/task_model.py
- env_wrappers/grid_coupled_env.py
- grid_model/grid_cache.py
- grid_model/OPF/MEF实现
- marl/bridges/*
- HARL MAPPO、GAE、buffer、loss和bounded Box分布实现
```

建议的实施顺序是先补metadata快照和可恢复代码版本，再开始正式长训练。是否实施上述最小修正，等待下一步确认。

---

## 附录：实际轻量验证

### 环境复现

- fresh `seed=7110` × 2：初始环境逐项比较。
- 同一合法动作序列 × 24步 × 2套fresh环境：物理、reward、state、cache逐步比较。
- `seed=7110`与`seed=7111`：server/task/obs/state和固定外部曲线比较。
- 同环境连续reset与fresh同seed reset：任务序列推进/恢复比较。
- probe/train/default eval/same-seed eval：对象与RNG隔离比较。

### 模型初始化

- 两次fresh runner、`seed=7110`、CPU、1 thread：Actor/Critic/optimizer/action比较。
- 一次fresh runner、`seed=7111`：只比较初始化hash，不执行update。

### 正式1-update命令模板

```powershell
$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:HARL_RUNTIME_PATH=(Join-Path (Get-Location) '.tmp_harl_runtime')
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'

& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' `
  -m train.train_harl_mappo_short `
  --config configs/harl_mappo_short.yaml `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path .tmp_harl_runtime `
  --seed 7110 `
  --updates 1 `
  --episode-length 24 `
  --rollout-threads 1 `
  --output-dir <独立临时目录> `
  --scenario idc_bess_padding `
  --device cpu
```

两次命令退出码均为0，各完成24 rollout步和1次update；未运行其他seed的第三次update，未运行5-update或40-update。
