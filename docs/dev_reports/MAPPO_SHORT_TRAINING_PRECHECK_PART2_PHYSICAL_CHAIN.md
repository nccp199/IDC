# MAPPO短训练前检查——第二部分：环境构造与原物理链继承

## 审计结论

结论选择：**2. 基本通过，需要少量包装或配置修正。**

正式双 Agent 环境完整复用了原单 Agent `IDCPriceEnv20D → GridCoupledEnv` 物理链。临时正式工厂等价性验证中，直接 `GridCoupledEnv` 与正式 HARL 环境的 17 个关键物理 `info` 字段最大绝对误差为 `0`；reward 仅因 HARL bridge 转为 `float32` 出现 `2.621576566941286e-09` 的表示误差。BESS 虚拟 21 维对 reward、SOC、功率、OPF、LMP、MEF、done 均无影响。

没有发现动作错位、reward 放大、state 被 padding 污染、绕过 GridCoupledEnv 或物理模型被复制重写的问题。但在正式实验前应处理四项可复现性/配置风险：

1. 启动校验在训练 VecEnv 上调用一次 `reset()`，runner warmup 再次 `reset()`；首个 task realization 被校验消耗。
2. `--scenario` 表示 HARL 协议/目录名，不表示项目物理 experiment case；物理 case 当前仍为 yaml 中的 `main`。
3. 一个 `seed` 同时作为算法、server 和 task seed，metadata 未分开记录三者。
4. 动态电网 CSV 加载失败会回退到固定 load scale；OPF/MEF 失败虽保留在 `info`，当前正式入口不会 fail-fast 或汇总失败率。

本次未修改任何环境、Bridge、配置或训练代码，未运行 policy update。

---

## A. 正式物理调用链

```text
train/train_harl_mappo_short.py::main()
→ resolve_config()
→ run_training()
→ marl/envs/harl_env_factory.py::make_harl_train_env()
→ _make_harl_vec_env()
→ make_harl_single_env()
→ configs/experiment_cases.py::get_experiment_case("main")
→ train/train_ppo_ultimate.py::make_unmonitored_env()
→ make_single_env()
→ data_io/data_loader.py::build_external_series_from_config()
→ envs/idc_price_env.py::IDCPriceEnv20D
→ env_wrappers/grid_coupled_env.py::GridCoupledEnv
→ marl/envs/idc_grid_multi_agent_env.py::IDCGridMultiAgentEnv
→ marl/bridges/harl_bridge.py::HarlIDCGridBridge
→ marl/bridges/harl_padded_bridge.py::HarlPaddedBridge
→ HARL harl/envs/env_wrappers.py::ShareDummyVecEnv
→ marl/runners/idc_mappo_runner.py::IDCOnPolicyMARunner
```

| 层级 | 文件路径与证据 | 类/函数 | 输入 | 输出 | 主要职责 |
| --- | --- | --- | --- | --- | --- |
| 正式入口 | `train/train_harl_mappo_short.py:364-469` | `run_training()` | resolved config、HARL path | runner lifecycle | 读取 `scenario`/`experiment_case`，构造正式 VecEnv并调用标准 runner |
| 正式工厂 | `marl/envs/harl_env_factory.py:17-52` | `make_harl_single_env()` | seed、HARL scenario、experiment case | `HarlPaddedBridge` | 只负责选择 case 与按固定顺序套 wrapper；不复制物理逻辑 |
| case选择 | `configs/experiment_cases.py:50-78` | `get_experiment_case()` | `main` | env/reward/data deep copy | `main` 不做额外覆盖；其他 case 只覆盖明示字段 |
| 公共物理工厂 | `train/train_ppo_ultimate.py:65-108` | `make_single_env()` | env/reward/data配置、seed | `GridCoupledEnv` | 合并配置、生成外部序列、设置 server/task seed、构造 base 和 grid wrapper |
| 无Monitor入口 | `train/train_ppo_ultimate.py:130-145` | `make_unmonitored_env()` | 同上 | 未套 SB3 Monitor 的 `GridCoupledEnv` | 与单 Agent 工厂共用同一 `make_single_env()` |
| IDC物理环境 | `envs/idc_price_env.py:32-283` | `IDCPriceEnv20D.__init__()` | 合并后的物理、reward、时序和seed参数 | 23维action、280维obs | 任务、服务器、能耗、成本、碳、BESS、PV、reward和episode状态 |
| IDC物理step | `envs/idc_price_env.py:437-954` | `IDCPriceEnv20D.step()` | `[IDC 22, BESS 1]` | obs、标量reward、terminated、truncated、info | 执行任务、功率/BESS/PV计算、reward与终止结算 |
| Grid包装 | `env_wrappers/grid_coupled_env.py:80-146` | `GridCoupledEnv.__init__()` | base env和grid配置 | action仍23维，obs 280→288 | 加载IEEE14、OPF/MEF、动态grid曲线、8维grid obs |
| Grid step | `env_wrappers/grid_coupled_env.py:216-274` | `step()` / `_run_grid_update()` | base transition | augmented obs、adjusted reward、扩展info | 使用 `P_bus_net_kW` 执行OPF/MEF并注入grid结果 |
| 多Agent层 | `marl/envs/idc_grid_multi_agent_env.py:29-82,228-260` | `IDCGridMultiAgentEnv` | `idc`/`bess`动作字典 | local obs字典、state、shared reward字典、done字典 | 仅拆分/重组动作观测并复制共享reward/done |
| HARL协议层 | `marl/bridges/harl_bridge.py:31-46,116-166` | `HarlIDCGridBridge` | 通用多Agent接口 | HARL异构tuple/array协议 | 固定Agent顺序、重复EP state、保留terminated/truncated到info |
| padding层 | `marl/bridges/harl_padded_bridge.py:76-105,129-190` | `HarlPaddedBridge` | obs `(288,164)`、action `(22,1)` | obs/action均齐次到22/288 | BESS obs尾部补零；BESS action只切第0维；state/reward/info不变 |
| VecEnv | `C:/Users/bulio/Desktop/IDC/HARL/harl/envs/env_wrappers.py:299-357` | `ShareDummyVecEnv` | 单个环境工厂 | batch维 `(1,...)` | stack结果；两个Agent都done时保存terminal obs/state并自动reset |

### A.1 关键确认

- 正式工厂不导入 `marl.tests`、pytest或测试helper。
- 物理创建逻辑只有 `make_single_env()` 一套；正式工厂复用 `make_unmonitored_env()`，没有复制IDC/Grid构造。
- 旧单Agent与正式MAPPO都从 `ENV_CONFIG/REWARD_CONFIG/DATA_CONFIG/GRID_*` 构造；差别是入口seed默认值和外层协议，不是物理模型。
- `IDCGridMultiAgentEnv` 强制底层action `(23,)`、grid obs 8维和既定字段契约，缺失时直接报错（`idc_grid_multi_agent_env.py:88-143`）。
- 正式工厂无法静默跳过 `GridCoupledEnv`：其构造顺序是直接代码路径；wrapper构造异常会由工厂包装为清晰 `RuntimeError`。
- 动态grid场景CSV加载是例外：`GridCoupledEnv._load_grid_scenario()` 捕获异常并回退（`grid_coupled_env.py:174-192`），当前入口未把 fallback 当作启动失败。

---

## B. 最终物理配置表

### B.1 scenario的两个含义

| 配置项 | 最终值 | 基础来源 | 场景覆盖 | CLI覆盖 | 运行时覆盖 |
| --- | ---: | --- | --- | --- | --- |
| HARL `env.scenario` | `idc_bess_padding` | `configs/harl_mappo_short.yaml:88` | 无物理覆盖 | `--scenario`可覆盖，但正式工厂只接受同一值 | 只用于协议校验、HARL task/log目录 |
| 项目物理 `experiment_case` | `main` | `configs/harl_mappo_short.yaml:90` | `get_experiment_case("main")`不覆盖基础配置 | 当前无CLI参数 | 工厂传入 `get_experiment_case()` |

明确答案：**`--scenario idc_bess_padding` 不选择物理实验场景。物理环境实际固定加载 resolved yaml 的 `experiment_case: main`。当前存在两个不同含义的 scenario 概念。**

### B.2 IDC、任务、服务器和外部时序

| 配置项 | 最终值 | 基础来源 | 场景覆盖 | CLI覆盖 | 运行时覆盖 |
| --- | ---: | --- | --- | --- | --- |
| horizon | `24`小时 | `configs/config_ultimate.py:25` | `main`无覆盖 | 无物理CLI | 无 |
| 算法/server/task seed | `7110/7110/7110` | MAPPO yaml seed；`make_single_env():71-81` | 无 | `--seed`同时改变三者 | server/task分别构造独立RNG，但seed数值相同 |
| server group | `20`组 | `IDC_SCALE_CONFIG:62` | 无 | 无 | `model.N=20` |
| 每组服务器数 | `100` | `IDC_SCALE_CONFIG:61` | 无 | 无 | effective server count=`2000` |
| seed 7110总算力 | `51955.13221781617` | server RNG + power model | 无 | seed间接改变 | 20组汇总 |
| seed 7110 idle/max功率 | `402.551895603204 / 1238.2892054708352 kW` | server RNG、group scale | 无 | seed间接改变 | 运行时抽样 |
| 单组最大task load | `0.60` | `ENV_CONFIG` | 无 | 无 | action前20维乘0.60 |
| task数量 | 随机任务30个 + 初始backlog任务1个 | `ENV_CONFIG num_tasks=30` | 无 | 无 | 每次reset重新生成 |
| 初始Q | 配置300，scale后`30000` workload | `ENV_CONFIG Q0=300`、`task_workload_scale=100` | 无 | 无 | `IDCPriceEnv20D:127` |
| task到达 | 随机任务到达小时1..18 | `task_model.py:207-270` | 无 | seed间接改变 | reset消耗task RNG |
| task类型概率 | inference 0.40；RL training 0.25；DL training 0.10；preprocess 0.25 | `task_model.py:78-118` | 无 | 无 | 每个task再随机duration/load/deadline/priority |
| task workload scale | `100` | `IDC_SCALE_CONFIG:63` | 无 | 无 | workload、Q/ref同时缩放 |
| price | 默认24h：0–6点0.35；7–9点0.65；10–14点1.05；15–17点0.65；18–20点1.05；21–23点0.65 | DATA path为None；`IDCPowerModel.create_price_curve()` | 无 | 无 | 无CSV覆盖 |
| carbon factor | 0–6点0.70；7–9点0.60；10–15点0.45；16–17点0.60；18–21点0.80；22–23点0.60 kgCO2/kWh | DATA path为None；`IDCPriceEnv20D:313-338` | 无 | 无 | 无CSV覆盖 |
| temperature | `25 + 5*sin(pi*(hour-8)/12)`，约20–30°C | DATA path为None；`IDCPriceEnv20D:223-227` | 无 | 无 | 无CSV覆盖 |
| workload/lambda曲线 | 每次reset由31个Task按arrival/workload构造 | `IDCPriceEnv20D:403-421` | 无 | seed间接改变 | 不是固定CSV曲线 |
| queue/lambda refs | queue=`600000`；lambda=`200000` | 基础ref × task scale100 | 无 | 无 | 构造时缩放 |

### B.3 reward与BESS/PV

| 配置项 | 最终值 | 基础来源 | 场景覆盖 | CLI覆盖 | 运行时覆盖 |
| --- | ---: | --- | --- | --- | --- |
| 基础reward权重 | done5.0；cost0.35；carbon0.30；SLA0.80；queue0.8；overflow1.2；final_queue3.0；deadline1.2；unused0.08；finished1.5；priority0.6；urgent0.8；waiting0.25；peak/grid_peak1.0；pause0.15；resume0.03；non_interruptible0.8；load_smooth0.05；action_smooth0.03；BESS degradation1.0；invalid0.2；final SOC2.0 | `configs/config_ultimate.py:86-110` | `main`无覆盖 | 无 | 直接传入base env |
| BESS容量 | 配置100 kWh；运行时`10000 kWh` | `ENV_CONFIG` + `bess_scale_factor=100` | 无 | 无 | `scale_bess_with_idc=True` |
| SOC | init0.50；min0.10；max0.90；target0.50；final tolerance0.05 | `ENV_CONFIG` | 无 | 无 | reset把SOC恢复0.50 |
| 充/放电上限 | 配置20/20 kW；运行时`2000/2000 kW` | `ENV_CONFIG` + BESS scale100 | 无 | 无 | 受SOC、效率和IDC负荷进一步约束 |
| 充/放电效率 | `0.95 / 0.95` | `ENV_CONFIG` | 无 | 无 | 无 |
| 退化 | throughput cost=`0.02/kWh` | `ENV_CONFIG` | 无 | 无 | 只有成本与累计throughput；**无SOH状态/演化模型** |
| PV | 默认bell曲线；capacity/ref=`500 kW`；12点峰值500；不允许export | `DATA_CONFIG:68-84`、`data_loader.py:104-137` | 无 | 无 | PV先抵扣BESS后本地净负荷，多余curtail |

### B.4 Grid

| 配置项 | 最终值 | 基础来源 | 场景覆盖 | CLI覆盖 | 运行时覆盖 |
| --- | ---: | --- | --- | --- | --- |
| IEEE系统 | pandapower IEEE14，baseMVA100 | `GRID_CONFIG`、`ieee14_loader.py:12-52` | 无 | 无 | 原始active/reactive load=`259 MW / 73.5 Mvar` |
| IDC接入 | IEEE bus 9；pandapower index 8 | `GRID_CONFIG:116` | 无 | 无 | 追加一个p_mw=`P_bus_net_kW/1000`的load |
| OPF | AC | `GRID_CONFIG:117` | 无 | 无 | 每步调用；cache命中时复用结果 |
| LMP | 无独立开关；随OPF提取bus `lam_p` | `opf_solver.py:187-205` | 无 | 无 | bus 9 LMP进入obs/info；grid reward权重为0 |
| MEF | 开启；delta_p=`0.1 MW` | `GRID_CONFIG` | 无 | 无 | 每步plus/minus OPF或cache |
| grid base load scale | 动态24h，`0.876997...`至`1.113496...` | `GRID_SCENARIO_CONFIG` | NEMS Singapore CSV | 无 | 当前成功加载绝对路径CSV |
| IDC load scale | 物理IDC侧server group scale100；送电网的增量是`P_bus_net_kW/1000` | IDC_SCALE + Grid step | 无 | seed影响IDC功率 | 动态grid scale只缩放IEEE原有负荷 |
| PV进入bus | `P_bus_net = P_IDC + P_charge - P_discharge - PV_used` | `IDCPriceEnv20D:590-598` | 无 | 无 | no-export下不小于0 |
| 电压限制 | 当前network各bus min=`0.94`；max=`1.06/1.07/1.09`（generator setpoint harmonization后） | pandapower case14 + `harmonize_generator_voltage_limits()` | 无 | 无 | metrics优先使用每bus limit；缺失才fallback 0.95/1.05 |
| 线路限制 | `100%` | IEEE14 line table | 无 | 无 | overload按每line limit判断 |
| OPF失败 | `OPFResult(success=False,message=...)`；grid obs无效数值归零，success位0 | `opf_solver.py:70-111,312-315` | 无 | 无 | info保留失败与message；不抛出训练 |
| MEF失败 | `MEFResult(success=False,message=...)` | `grid_coupled_env.py:326-345` | 无 | 无 | info保留失败与message；obs MEF归零 |
| grid reward | 关闭、mode=`none`、三个权重全0 | `GRID_REWARD_CONFIG` | 无 | 无 | adjusted reward等于base reward |
| cache | 开启OPF/MEF；bin 0.1MW/0.005；max50000；reset不清；per_worker；不缓存失败 | `GRID_CACHE_CONFIG` | 无 | 无 | 本部分仅记录，不审计性能/近似误差 |

---

## C. 动作语义表

| 层级 | 输入shape | 输出shape | IDC动作 | BESS动作 | 是否改值 |
| --- | ---: | ---: | --- | --- | --- |
| IDC Actor | obs batch→action | 每线程`(22,)` | 20组执行强度 + urgent preference + continuity preference | 无 | HARL bounded Box直接采样到`[0,1]` |
| BESS Actor | padded obs→action | 每线程`(22,)` | 无 | 第0维有效，1:22为虚拟策略维 | bounded Box采样；无Bridge clip |
| runner collect | 两个actor | `(n_threads,2,22)` | agent0完整22维 | agent1完整22维 | 只stack/transpose |
| `HarlPaddedBridge` | `(2,22)` | `(22,)`和`(1,)` | `actions[0]`完整复制 | `actions[1,:1]`；虚拟21维丢弃 | slice，不clip、不缩放（`harl_padded_bridge.py:143-164`） |
| `HarlIDCGridBridge` | ordered 2-element sequence | `{"idc":(22,),"bess":(1,)}` | index0→idc | index1→bess | 只换容器（`harl_bridge.py:102-114`） |
| `IDCGridMultiAgentEnv` | 动作dict | flat `(23,)` | 前22维 | 最后1维 | FlatActionAdapter严格检查后concat |
| `FlatActionAdapter` | `(22,)`,`(1,)` | `(23,)` | 原值 | 原值 | 不clip；越界直接报错（`action_adapter.py:24-54`） |
| `GridCoupledEnv` | `(23,)` | 原样转交base | 原值 | 原值 | 不修改动作（`grid_coupled_env.py:216-218`） |
| `IDCPriceEnv20D` | `(23,)` | 物理控制量 | 0:20×0.60形成计划task load；20/21为偏好 | index22映射`2a-1` | 先flatten并`np.clip([0,1])`；正式上游已严格有界，因此合法路径数值不变 |

### C.1 动作不变量

- agent0的22维全部成为flat action前22维，没有slice或重排。
- agent1只有第0维成为flat action第22维；虚拟21维在进入MultiAgentEnv前已丢弃。
- 拼接顺序始终是 `[IDC 22维, BESS 1维]`。
- BESS normalized action：`0.0 → raw -1 → 最大充电意图`；`0.5 → raw 0 → idle`；`1.0 → raw +1 → 最大放电意图`。
- 实际BESS功率还受SOC、charge/discharge效率、功率上限和放电不超过 `P_IDC_kW` 的物理约束。
- 底层 `IDCPriceEnv20D:462` 存在历史安全clip；它不是正式训练的有界分布替代品，且本次合法动作验证中没有改变数值。

---

## D. 观测与状态表

| 层级 | IDC obs | BESS obs | state | 处理 |
| --- | ---: | ---: | ---: | --- |
| `IDCPriceEnv20D` | 原始280 | — | — | 6 global + 10 task pool + 6×20 server + 6×24 forecast |
| `GridCoupledEnv` | 288 | — | — | 在280尾部追加8维grid features |
| `IDCGridMultiAgentEnv` | 288 | 164 | 294 | IDC复制完整288；BESS取6 global +144 forecast +8 grid +6 supplemental；state=完整288+6 supplemental |
| `HarlIDCGridBridge` | 288 | 164 | 每Agent相同294 | 只按固定顺序转array；state重复2份 |
| `HarlPaddedBridge` | 288 | 288 | 每Agent相同294 | BESS 164后追加124个float32零；state不padding |
| `ShareDummyVecEnv` | `(1,2,288)` | 同batch | `(1,2,294)` | 只增加env batch维 |

### D.1 原始280维字段顺序

```text
0:6       global = T、price、lambda、Q、time_sin、time_cos
6:16      task pool = waiting/running/finished/unfinished、urgent/overdue、deadline、priority、parallelizable、interruptible
16:136    server groups = prev_load、capacity、efficiency、unit cost、available、temperature，各20维
136:280   full-horizon forecast = price、temperature、lambda、PV、time_sin、time_cos，各24维
```

Grid追加的8维顺序（`grid_coupled_env.py:347-375`）：

```text
280:288 = LMP、MEF+、MEF-、normalized min voltage、max line loading、network loss、security penalty、OPF success
```

6维supplemental顺序（`marl/specs/agent_specs.py`）：

```text
bess_soc、bess_energy_kWh、P_IDC_kW、P_grid_kW、bess_charge_power_kW、bess_discharge_power_kW
```

### D.2 继承结论

- BESS补零只发生于 `HarlPaddedBridge._pad_observations()`，不会传入动作或底层物理环境。
- state 294不是两个padded obs拼接，而是 `GridCoupledEnv`真实288维观测加6维物理supplemental。
- state不含124个padding零；两个Agent收到相同EP state。
- state/obs包含全局信息以及已知/可预测的完整24h price、temperature、task arrival、PV和时间编码；不包含未来action导致的queue、完成量或服务器真实负荷。
- 临时验证 reset/step 均为有限值；shape分别为obs `(1,2,288)`、state `(1,2,294)`。
- OPF/MEF失败时无效grid数值在obs中转为0，原始NaN和失败message仍保留在info。

---

## E. reward与done链

| 层级 | reward shape/类型 | done语义 | 数值处理 |
| --- | --- | --- | --- |
| `IDCPriceEnv20D` | Python `float`标量 | 第24步`terminated=True`；`truncated=False` | 计算base reward；终端加入final queue和SOC penalty |
| `GridCoupledEnv` | Python `float`标量 | 原样传递terminated/truncated | `adjusted=base-grid_penalty`；当前grid reward关闭，数值完全等于base |
| `IDCGridMultiAgentEnv` | `{"idc":r,"bess":r}` | 分别复制给两Agent和`__all__` | 同一shared scalar复制，不平均、不求和、不缩放 |
| `HarlIDCGridBridge` | float32 `(2,1)` | `done=terminated or truncated`，shape `(2,)` | reward按Agent顺序复制；info保留两个原flag并设`bad_transition=truncated` |
| `HarlPaddedBridge` | `(2,1)` | `(2,)` | 原样传递 |
| `ShareDummyVecEnv` | `(1,2,1)` | `(1,2)` | stack；两Agent都done时自动reset，terminal state存入agent0 info |
| HARL EP critic buffer | `(1,1)` | masks由all-agent done生成 | `OnPolicyBaseRunner.insert():448-456`明确取`rewards[:,0]`，不会把shared reward相加两次 |

明确答案：**相同seed、相同初始task realization和相同23维物理动作下，直接GridCoupledEnv与正式双Agent环境的底层标量reward应相等。** 临时验证的物理reward计算相同，HARL数组float32表示误差最大为`2.62e-9`。

当前标准MAPPO链未进入Safe/Lagrangian算法。`safe_violation_*`/`safe_cost_*`只是Grid info字段；grid reward关闭，安全字段不改变reward。

### E.1 episode/reset

- `environment horizon=24`与`episode_length=24`严格对齐。
- 第1–23步done为false；第24步两Agent done为true。
- 终端reward在返回前已经包含final queue/SOC结算，不存在未结算终端reward。
- 第24步实测：`terminated=True`、`truncated=False`、`bad_transition=False`。
- VecEnv在第24步后自动reset；实测底层 `current_step=0`。
- base reset恢复task队列、Q、prev action/load、BESS SOC/energy和累计指标；Grid reset执行nominal zero-IDC-load OPF并保留cache（当前reset不清cache）。
- terminated/truncated在Bridge合成HARL done，但原语义保留在每Agent info；HARL bad mask读取agent0 `bad_transition`，time-limit语义未丢失。

---

## F. info字段传递表

正式VecEnv第1步实测 `infos.shape == (1,2)`；每个Agent info含238个键。GridCoupledEnv先复制base info再追加grid字段；MultiAgentEnv继续复制；Bridge为两个Agent各复制一次并加入 `agent_id/terminated/truncated/bad_transition`；PaddedBridge不改；VecEnv保留为二维object array。

| 类别 | 字段/别名 | 底层产生位置 | Grid保留 | MultiAgent保留 | Bridge保留 | VecEnv最终存在 |
| --- | --- | --- | --- | --- | --- | --- |
| 任务完成 | `completed_work`, `total_completed_work`, `finished_task_count`, `newly_finished_count`, `task_completion_rate`, `completion_rate` | IDC step/task metrics | 是 | 是 | 两份 | 是 |
| backlog/queue | `Q`, `backlog_work`, `final_backlog_work`, `overflow_work`, `urgent_backlog_work`, `unfinished_task_count` | IDC step | 是 | 是 | 两份 | 是 |
| deadline/SLA | `new_deadline_miss_count`, `deadline_miss_count/rate`, `sla_penalty`, `sla_violation_count/rate`, `avg/max_task_delay` | IDC step | 是 | 是 | 两份 | 是 |
| task数量 | `total_task_count`, `paused_task_count`, waiting/turnaround相关字段 | IDC reset/step/task metrics | step字段是；reset info见下述例外 | 是 | step两份 | step存在 |
| IDC功率 | `P_IDC`, `P_IDC_kW`, `P_IT`, `P_cooling`, `PUE`, `COP`, `idc_peak_power_kW` | IDC step | 是 | 是 | 两份 | 是 |
| grid功率/能量 | `P_bus_net_kW`, `P_grid_kW`, `grid_power_kW`, `grid_energy_kWh`, `energy_kWh`, `total_grid_energy_kWh` | IDC step | 是 | 是 | 两份 | 是 |
| 成本/电价/峰值 | `cost/hourly_cost`, `total_cost`, `price`, `grid_peak_power_kW`, `peak_power_kW`, excess/episode/total字段 | IDC step | 是 | 是 | 两份 | 是 |
| 碳 | `carbon_factor`, `carbon_emission`, `total_carbon_emission`, `carbon_cost`, `carbon_per_task` | IDC step | 是 | 是 | 两份 | 是 |
| Grid排放/MEF | `grid_total_emission_kg`, `grid_mef_plus/minus`, `grid_mef_success/message`, `grid_mef_carbon*` | Grid wrapper | 新增 | 是 | 两份 | 是 |
| BESS SOC/功率 | `bess_soc`, `bess_energy_kWh`, `bess_charge/discharge_power_kW`, desired/available字段 | IDC step | 是 | 是 | 两份 | 是 |
| BESS动作 | `bess_raw_action`, `bess_mode`, `invalid_bess_action` | IDC step | 是 | 是 | 两份 | 是；normalized输入本身未作为info键保存 |
| BESS throughput/退化 | `bess_charge/discharge_kWh`, `bess_cycle_throughput_kWh`, totals, `bess_degradation_cost` | IDC step | 是 | 是 | 两份 | 是；无SOH字段 |
| PV | `PV`, `pv_available/used/curtail_kW/kWh`, totals, `pv_utilization_rate`, `renewable_share` | IDC step | 是 | 是 | 两份 | 是 |
| OPF | `grid_opf_success/message/mode`, case、bus、total load/generation/cost | Grid wrapper | 新增 | 是 | 两份 | 是 |
| 电压 | `grid_min/max_voltage_pu`, `grid_voltage_violation_count/magnitude` | Grid wrapper | 新增 | 是 | 两份 | 是 |
| 线路/损耗/LMP | `grid_max_line_loading_percent`, overload count/magnitude, `grid_network_loss_mw`, `grid_lmp` | Grid wrapper | 新增 | 是 | 两份 | 是 |
| grid场景/cache | source/message/load scale/USEP、OPF/MEF cache hit/size/rate | Grid wrapper | 新增 | 是 | 两份 | 是 |
| base reward | `reward_total`, 所有`r_*` components | IDC step | 是 | 是 | 两份 | 是 |
| grid reward | `base_reward`, `grid_reward_penalty`, `grid_adjusted_reward`, LMP/MEF/security penalty components | Grid wrapper | 新增 | 是 | 两份 | 是 |
| HARL终止 | `agent_id`, `terminated`, `truncated`, `bad_transition` | Bridge | — | — | 新增 | 是 |

### F.1 info例外与风险

- `ShareDummyVecEnv.reset()`不返回info，因此reset info不直接进入runner；可通过Bridge `last_info`读取。
- terminal时VecEnv只在 `infos[i][0]` 加入 `original_obs/original_state/original_avail_actions`，第二Agent info没有这三个键；HARL EP buffer和bad mask使用agent0，当前流程可工作。
- OPF/MEF失败字段不会被包装吞掉，但正式runner未主动读取/统计这些字段，训练可在失败后继续。
- `grid_reference_usep`只是场景参考info，当前不替换IDC `price_t`，也不进入base cost/reward。

---

## G. 测试结果

### G.1 现有parity测试

实际命令：

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -m pytest `
  marl/tests/test_multi_agent_parity.py `
  marl/tests/test_harl_bridge_parity.py `
  marl/tests/test_harl_padding_parity.py `
  marl/tests/test_harl_padding_action_invariance.py -q
```

结果：

- 退出码：`0`
- `4 passed`
- `0 failed`
- `225 warnings`
- 耗时：`44.12s`
- warnings：224个pandapower deprecation warning + 1个pytest cache权限warning
- 是否直接覆盖正式环境工厂：**否**。四个测试通过 `marl/tests/helpers.py::make_grid_env()` 构造 `main` case；其物理创建函数与正式工厂相同，但没有调用 `make_harl_train_env()`。

### G.2 临时正式工厂物理等价性验证

实际执行方式：

```powershell
$env:PYTHONPATH=(Get-Location).Path + ';C:\Users\bulio\Desktop\IDC\HARL'
$env:PYTHONDONTWRITEBYTECODE='1'
@'
# 内联只读验证：构造直接GridCoupledEnv、正式make_harl_train_env、
# 两个正式虚拟动作环境、旧测试helper环境和24步正式episode。
'@ | & 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -
```

脚本未写入项目文件，未创建actor/critic，未运行policy update。

| 验证 | 结果 | 正式工厂直接覆盖 |
| --- | --- | --- |
| A：直接GridCoupledEnv vs正式工厂，3步、17个物理字段 | 通过；物理info最大误差`0`；reward最大误差`2.621576566941286e-09` | 是 |
| B：固定IDC和BESS第0维，只改虚拟21维 | 通过；reward/done/全部选定物理字段逐值完全相等 | 是 |
| C：旧测试helper vs正式工厂 | 通过；reward最大误差`1.5359686647564708e-09` | 是 |
| 24步episode | 仅第24步done；terminated true；truncated/bad_transition false；自动reset到step0 | 是 |
| reset/step shape与有限值 | obs `(1,2,288)`、state `(1,2,294)`、reward `(1,2,1)`、done `(1,2)`，均有限 | 是 |
| info结构 | `(1,2)`，单Agent 238键 | 是 |
| double reset task序列 | fresh env首序列可复现；同一env第1/第2次reset不同 | 是 |

运行结果：

- 退出码：`0`
- 失败检查：`0`
- warnings：`0`
- 总耗时：`66.5s`
- 物理字段容差：`1e-8`
- reward容差：`1e-7`，原因是Bridge明确转为float32；实测误差远小于容差

---

## H. 问题清单

### 阻塞MAPPO短训练

- **当前main物理链没有阻塞项。** 正式环境可完成完整24步episode，动作/reward/state/done与直接物理环境等价。

### 必须在正式实验前修正

1. **启动校验消耗首个task realization。**
   - 证据：`validate_environment_contract()`在训练VecEnv上调用reset（`train_harl_mappo_short.py:323`）；随后HARL `warmup()`再次reset（`on_policy_base_runner.py:269-276`）。
   - 实测：同seed fresh env的第一次reset完全相同，但同一env第一次与第二次reset不同。
   - 影响：训练不是从seed对应的第一个task realization开始；校验逻辑改变训练随机状态。

2. **HARL scenario与物理case名称混用。**
   - `--scenario idc_bess_padding`只影响协议校验和HARL目录；物理仍是yaml `experiment_case: main`。
   - CLI没有 `--experiment-case`；metadata只突出 `scenario`，容易把HARL命名误认为物理场景。

3. **算法/server/task seed耦合且记录不足。**
   - `make_single_env()`把同一worker seed同时写入 `server_seed` 和 `task_seed`。
   - 这在技术上可复现，但把网络初始化随机性、服务器物理抽样和task arrival随机性绑定；当前run metadata未分开记录。

4. **动态grid场景和OPF/MEF缺少正式启动保障。**
   - 当前CSV真实加载成功；但文件缺失/解析失败时会回退到`load_scale=1.0`，正式入口不会阻止训练。
   - OPF/MEF失败会进入info并把obs特征置0，但grid reward关闭且runner不汇总失败，可能在训练日志层面静默。

### 建议改进

- 启动校验直接核对底层实际 `horizon/grid_enabled/opf_mode/use_mef/grid_scenario_source`，而不只校验shape和yaml值。
- metadata显式记录 `experiment_case`、algorithm seed、environment seed、server seed、task seed、动态grid source/message。
- 在正式物理日志阶段汇总OPF/MEF成功率和fallback次数；是否fail-fast应成为明确配置，不应隐式。
- 若后续需要动作诊断，在info补充normalized BESS input；当前只有`bess_raw_action`和实际功率。
- 文档固定说明VecEnv info是`(n_envs,n_agents)`嵌套结构，以及terminal original state只写入agent0 info。

### 可以保持现状

- 复用 `make_unmonitored_env()/make_single_env()`，不复制物理模型。
- Agent顺序 `0=IDC,1=BESS`，动作拼接 `[IDC22,BESS1]`。
- BESS虚拟21维在物理入口前丢弃。
- shared reward复制给两个actor、critic只取agent0，不放大2倍。
- BESS local obs尾部补零和294维EP state设计。
- Grid reward保持关闭——前提是短训练目标确实只观察grid而不把LMP/MEF/security另加到reward。
- 底层合法动作路径上的历史clip；正式bounded Box和FlatActionAdapter已在上游保证边界。
- `GridCoupledEnv.reset()`使用nominal zero-IDC-load grid observation，已有 `grid_initial_obs_mode` 明示。
- train/eval工厂每次分别创建独立wrapper和底层环境对象，不共享物理状态或cache。
- 正式工厂当前不注入BESS monitor；padding层即使以后注入monitor，也是在物理transition完成后只读记录，不改变动作或返回值。

---

## I. 第二部分判断

**2. 基本通过，需要少量包装或配置修正。**

物理转移、动作语义、reward、done、obs/state和info继承已经通过静态追踪与正式工厂实测。剩余问题集中在启动校验改变task RNG状态、场景命名、seed可审计性和grid降级保障，不需要修改 `IDCPriceEnv20D`、`GridCoupledEnv`物理公式、Bridge、padding或MAPPO算法。

---

## J. 最小修正建议（本次未实施）

```text
建议修改：
- train/train_harl_mappo_short.py
  - 用独立probe环境执行启动reset校验，或校验后重新构造干净train环境；不要消耗训练env首个task序列。
  - metadata分开记录harl_scenario、experiment_case、algorithm/env/server/task seed。
  - 启动时校验实际horizon、grid_enabled、opf_mode、use_mef和grid_scenario_source不是fallback。

- configs/harl_mappo_short.yaml
  - 明确区分harl_scenario与experiment_case。
  - 增加可审计的environment/server/task seed策略字段；默认值可继续相等，但必须显式。
  - 增加是否允许grid scenario fallback、是否对OPF/MEF失败fail-fast的配置声明。

- marl/envs/harl_env_factory.py
  - 接收明确的environment/server/task seed并传入公共make_single_env链。
  - 提供只读physical contract描述，供正式入口校验；不复制或修改物理逻辑。

建议新增：
- marl/tests/test_harl_env_factory_physical_parity.py
  - 将本次临时A/B/C验证固化为正式工厂回归测试。
  - 覆盖首个task序列不被启动校验消耗、grid source非fallback和24步episode对齐。

保持不动：
- envs/idc_price_env.py
- env_wrappers/grid_coupled_env.py
- idc_model/task_model.py
- idc_model/power_model.py
- grid_model/**
- marl/adapters/action_adapter.py
- marl/bridges/harl_bridge.py
- marl/bridges/harl_padded_bridge.py
- HARL MAPPO/GAE/buffer/bounded Box实现
```

---

## K. 审计上下文与函数级证据

本节记录底层审计模型，避免结论只停留在文件名推断。

### K.1 全局不变量

1. 任一合法正式动作必须保持 `agent0[22] + agent1[0] → flat[23]`，顺序不可改变。
2. 任一物理transition只允许调用一个 `GridCoupledEnv.step(flat23)`，wrapper不得提前读取未来transition功率。
3. shared reward对两个actor相同，EP critic只消费一份；任何求和都会造成2倍放大。
4. centralized state必须是 `raw wrapped obs 288 + physical supplemental 6`，不能由padded local obs拼接。
5. terminated与truncated可在HARL done中合并，但必须通过info/bad mask保留time-limit区别。
6. 物理experiment case、HARL scenario和grid动态scenario是三个独立概念。

### K.2 `make_single_env()`微分析

**Purpose：** `train/train_ppo_ultimate.py:65-108`是单Agent与正式MAPPO共同的唯一物理构造点。它决定配置合并顺序、外部时序、server/task seed和GridCoupledEnv是否存在，因此是物理继承的信任根。

**Inputs & Assumptions：** env/reward/data dict来自deep-copied experiment case；seed是可信整数或None；rank当前为0；IDC_SCALE_CONFIG和GRID_*是模块级正式配置；外部CSV路径必须可读；所有传给IDC构造器的键必须唯一兼容。

**Outputs & Effects：** 创建新的IDC task/power模型；创建新的IEEE14 Grid wrapper/cache；seed action/obs spaces；monitor=False时直接返回GridCoupledEnv；不写物理文件。

**Block-by-Block：**

- L76生成worker seed。为什么在最前：server/task模型构造必须在随机抽样前得到seed；假设rank稳定。
- L77-84按 `env → reward → IDC scale → external series → seed` 合并。第一性原理：物理行为完全由最终kwargs决定；同名后项覆盖前项，因此scale和seed是运行时覆盖。
- L85实例化IDC。为什么先于Grid：Grid需要base action/obs/horizon属性。
- L86-92实例化Grid。5 Hows：base物理transition产生bus net load，Grid读取该字段，OPF/MEF追加反馈，action space保持base不变，最终包装返回。
- L93-95只seed spaces，不重置task RNG；task RNG已在模型构造时用worker seed初始化。

**Cross-Function Dependencies：** 调用data loader、IDC构造器和Grid构造器；被单Agent、测试helper和正式HARL factory共同调用。风险点是配置merge覆盖、CSV fallback和一个seed绑定两类物理RNG。

### K.3 `IDCPriceEnv20D.reset()/step()`微分析

**Purpose：** reset定义episode初始task/BESS/累计状态，step实现唯一IDC/BESS物理transition。wrapper必须把相同flat23动作送达此处，才能声明物理等价。

**Inputs & Assumptions：** action必须23维、数值有限且理论上在[0,1]；current_step小于24；task RNG已初始化；price/carbon/temp/PV曲线均长24且有限；BESS参数有效；task对象状态属于当前episode。

**Outputs & Effects：** reset重新生成task并恢复SOC/计数；step推进task、BESS、能耗/成本/碳、queue和时间；返回280维obs、标量reward、两个终止flag和完整info；第24步结算final penalties。

**Block-by-Block：**

- reset L370-401恢复全部运行状态。为什么在task生成前：旧episode状态不得污染新task realization。
- reset L403-421使用持续task RNG生成30个随机task并插入初始backlog。5 Whys：连续reset为何不同——RNG未重建；为何fresh env相同——构造seed相同；为何校验会影响训练——校验先消费一次；为何shape校验看不出——shape不依赖task内容；为何必须修正——实验seed语义应独立于校验副作用。
- step L456-468 flatten、维度检查、clip和动作切片。合法正式链已在上游拒绝越界，底层clip只作为最后防线。
- L470-525把IDC 22维转为计划capacity、执行task并反推实际负荷/功率。依赖task状态和server参数。
- L527-609把BESS normalized动作映射到充放电，执行SOC/功率限制，再用PV抵扣，形成bus net/grid purchase、成本和基础碳。
- L630-733计算所有base reward components。Grid LMP/MEF不在这里；碳reward使用固定carbon factor。
- L735-763在第24步终止、加入final queue/SOC penalty、更新内部状态并给终端零base obs。
- L796-954构造info；后续Grid和所有MARL wrapper只复制/追加。

**Cross-Function Dependencies：** 调用task execution、power/PUE模型和observation builders；被GridCoupledEnv唯一包装调用。风险是底层clip掩盖非正式调用者越界、连续reset推进task RNG、SOH未建模但不影响当前既定语义。

### K.4 `GridCoupledEnv.step()/_run_grid_update()`微分析

**Purpose：** 把已完成的IDC/BESS/PV transition映射到IEEE14反馈。它不改变base action或done，只追加grid observation/info，并按配置可选修改reward。

**Inputs & Assumptions：** base info包含有限 `P_bus_net_kW`和hour；IEEE14 case有效；dynamic load scale非负；OPF/MEF返回结构完整；grid reward配置明确；cache键量化符合当前实验约定。

**Outputs & Effects：** 每步求OPF和可选MEF；追加8维grid obs与grid info；当前返回原base reward；更新内存cache统计。

**Block-by-Block：**

- step L218先调用base。为什么必须先：BESS/PV后的真实bus net load只有base transition完成后才存在。
- L226-234读取 `P_bus_net_kW`，转换MW并调用grid update。变量名`idc_load_mw`实际是bus net load，包含BESS/PV影响。
- `_run_grid_update` L239-269选择动态scale、OPF、metrics/emission和MEF，再注入info。5 Hows：IDC功率→BESS/PV净化→MW增量load→IEEE原负荷按小时scale→OPF/LMP/MEF。
- L270-274计算可选grid penalty。当前enabled false/mode none/weights0，所以严格返回base reward。
- `_load_grid_scenario` L174-192捕获所有异常并fallback。风险：物理scenario可降级但训练不中止；仅source/message暴露。
- MEF L326-345捕获异常并返回failed result；OPF solver也返回failed result。风险不是信息丢失，而是runner不消费失败状态。

**Cross-Function Dependencies：** 调用OPF solver、MEF calculator、grid metrics、emission和cache；被MultiAgentEnv直接包装。与state builder耦合，因为8维grid obs进入两local obs和central state。

### K.5 `IDCGridMultiAgentEnv.step()`与动作adapter微分析

**Purpose：** 把一个legacy 23维控制面拆成两个Agent接口，同时保持底层transition单一。它还从同一raw obs构造两个local obs和一个global state，并把标量reward/done共享给两Agent。

**Inputs & Assumptions：** action dict必须恰有idc/bess键；IDC 22维、BESS 1维、有限且[0,1]；底层是启用8维grid obs的GridCoupledEnv；transition info含6个supplemental字段；底层reward有限。

**Outputs & Effects：** concat成flat23并调用一次grid step；保存last raw obs/action/info；构造local obs/state；复制reward和终止flag；不改变底层环境状态之外的物理量。

**Block-by-Block：**

- adapter L24-37严格校验，越界报错不clip。为什么在concat前：可把错误归因到具体Agent。
- adapter L39-54只concat。第一性原理：保持legacy transition等价要求元素和值逐一不变。
- multi step L230-235校验exact keys后调用一次底层step；没有第二条物理路径。
- L238-242只在transition完成后保存info并构造下一状态，避免提前读取功率。
- L244-259复制同一reward和done；没有sum/mean/scale。

**Cross-Function Dependencies：** 调用FlatActionAdapter、三个observation/state builders和GridCoupledEnv；被HARL Bridge调用。主要风险是supplemental字段schema变化会fail-fast，属于显式契约而非静默降级。

### K.6 HARL Bridge、padding与VecEnv微分析

**Purpose：** Bridge把字典协议转为固定Agent顺序；padding只解决HARL stack要求；VecEnv增加batch并自动reset。三层都不得改变物理动作、reward或state语义。

**Inputs & Assumptions：** agent order固定 `(idc,bess)`；真实obs/action分别 `(288,164)/(22,1)`；state `(294,)`；padded输入 `(2,22)`；两个Agent共享终止；info是可复制dict。

**Outputs & Effects：** Bridge生成obs tuple、重复state、reward `(2,1)`和done `(2,)`；padding输出obs `(2,288)`、slice BESS动作；Vec输出带env batch的数组并在terminal自动reset。

**Block-by-Block：**

- Bridge action L102-114按index映射固定Agent。为什么不用mapping：HARL runner提供ordered agent axis。
- Bridge step L140-157合并done但把原flag写回两个info，`bad_transition=truncated`维持GAE time-limit mask。
- padding L144-153复制IDC、保留BESS full用于diagnostics、只取BESS第0维进入物理链。虚拟维不可能流入FlatActionAdapter。
- padding L170-181的monitor只在底层transition返回后记录；formal factory当前传`diagnostics=None`，且即使启用也不修改result。
- Vec step_wait L319-348只stack；`np.all(done)`时先保存terminal数据再reset。5 Whys：为何24步返回新obs——Vec自动reset；为何terminal reward仍正确——reward/info来自reset前result；为何current_step变0——reset已执行；为何buffer done仍true——dones未替换；为何GAE可断episode——runner masks使用原dones。

**Cross-Function Dependencies：** Bridge依赖MultiAgent contract；padding依赖Bridge真实space；Vec依赖两个Agent都done；HARL EP critic依赖agent0 state/reward/info。风险集中在info嵌套和启动额外reset，不在动作/reward物理变换。

---

## L. 审计限制

- 本部分没有深入评估cache量化对长期训练的近似误差。
- 没有运行MAPPO actor/critic、5-update或40-update训练。
- 没有修改或新增测试文件；正式工厂等价性使用不落盘的临时内联脚本。
- 未验证所有可能experiment case；本结论针对正式resolved配置 `experiment_case=main`。
- OPF/MEF在抽样动作与24步中真实执行，info传递路径已验证；本次没有系统汇总全episode成功率，也未故意构造失败注入场景。
