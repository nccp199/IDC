# 当前科研项目仓库：面向论文写作的实现审计

审计日期：2026-08-10  
项目：基于微电网的智算中心协同调控  
审计范围：当前工作区最新代码，不修改算法、环境、配置或实验逻辑  

> 本报告只陈述代码事实，不评价创新水平，不替现有方法包装创新。报告生成完全基于本地仓库，不依赖网络。

## 1. 审计口径与版本边界

本次审计沿正式调用链检查了环境、双智能体接口、MAPPO、HAPPO、HGTA、集中式 critic、actor、观测、动作、reward、constraint、电网、AC OPF、IDC、任务、PV、BESS、checkpoint/resume、evaluation、seed、logging、metrics、graph construction 和 graph attention。

证据等级：

- **已实现且有验证证据**：当前代码存在，并有本次测试或仓库内成功运行产物支持。
- **已实现但尚未充分验证**：代码链完整，但没有覆盖正式规模、多 seed 或统计性能的证据。
- **接口/预留**：存在字段、配置或输出，但未进入正式控制或训练逻辑。
- **未实现**：当前正式链中不存在。

版本注意事项：项目 Git HEAD 为 `e3e77c487a1e59ed6d891e5ae34fe4732c580754`，但当前工作树是 dirty 状态，且正式 MARL/HGTA 相关大量文件仍为 untracked。HARL 位于同级目录 `C:\Users\bulio\Desktop\IDC\HARL`，HEAD 为 `050ad6a294fe9f7572985dea910d59ea6d4f94b4`，工作树干净。故本报告的事实基线是“当前工作区”，不能仅靠项目 Git commit 完整重建。

核心入口与配置：

- `configs/harl_mappo_short.yaml`
- `train/train_harl_mappo_short.py::run_training`
- `marl/envs/harl_env_factory.py::make_harl_train_env`
- `train/train_ppo_ultimate.py::make_single_env`
- `marl/runners/idc_mappo_runner.py::IDCOnPolicyRunner`
- `eval/eval_harl_mappo_fixed.py`

## 2. 当前系统结构图（文字版）

```text
随机生成 Task + 初始 backlog Task
  -> 到达激活与 Task.remaining_work 队列
  -> IDC Agent 输出：20 个 server-group 强度 + urgent 偏好 + continuity 偏好
  -> 聚合可用算力；按任务得分顺序分配工作量
  -> 根据实际完成量反推 20 组实际负载
  -> IT 功耗 + 制冷功耗 + 其他功耗 = P_IDC
  -> BESS Agent 输出 1 个充放电标量；PV 为外生不可控出力
  -> P_bus_net = P_IDC + P_charge - P_discharge - P_PV_used
  -> 不允许上网，P_grid = max(P_bus_net, 0)
  -> P_grid 作为额外有功负荷接入 IEEE-14 的 IEEE bus 9
  -> 每小时 AC OPF（或命中 cache）+ 0.1 MW 有限差分 MEF
  -> 生成 LMP、MEF、电压、线路负载、损耗、OPF success 等
  -> 8 维网架观测追加到下一时刻 observation
  -> 环境基础 reward 在 OPF 前按任务/购电/碳/峰值/BESS 等计算
  -> 正式配置关闭 grid reward，OPF 指标不直接改变 reward
  -> 两个 agent 收到完全相同的 shared team reward
```

真实调用链：

```text
train_harl_mappo_short.py
  -> prepare_training_environment
  -> make_harl_train_env
  -> get_experiment_case("main")
  -> train_ppo_ultimate.make_single_env
  -> IDCPriceEnv20D
  -> GridCoupledEnv
  -> IDCGridMultiAgentEnv
  -> HarlIDCGridBridge
  -> HarlPaddedBridge
  -> ProjectShareSubprocVecEnv
  -> IDCOnPolicyRunner
```

### 九个耦合问题的直接答案

| 问题 | 当前代码事实 |
|---|---|
| 1. IDC 负荷如何产生 | IDC actor 的 20 个动作先形成计划任务负载与聚合算力；任务按 urgent/continuity 得分顺序消耗算力；环境按实际完成量和 40% 未用计划负载预留损耗反推实际 server-group load，再进入功耗模型。 |
| 2. PV 如何进入 | PV 是外生时序。先计算 `P_IDC + charge - discharge`，再以 `min(PV_available, positive_local_demand)` 就地消纳；剩余为弃光，不允许反送电。 |
| 3. BESS 如何进入 | 能源 agent 的一个标量映射到 `[-1,1]`；负值充电、正值放电。功率、SOC、效率和不得超出 IDC 负荷均由环境硬裁剪。 |
| 4. IDC/PV/BESS 与 IEEE 电网在哪里耦合 | 三者先在单一 PCC 聚合为 `P_bus_net/P_grid`，然后作为 `q_mvar=0` 的额外有功负荷接入 pandapower index 8，即 IEEE bus 9。BESS/PV 没有作为独立 storage/sgen 元件进入 OPF。 |
| 5. AC OPF 在哪一步调用 | `GridCoupledEnv.step` 在底层 IDC/BESS step 完成后调用；reset 时还用零 IDC 负荷初始化一次。cache miss 时执行 `pandapower.runopp`。 |
| 6. OPF 输出 | 发电出力、总发电成本、bus `lam_p`、bus 电压、线路 loading、总负荷、总发电、网损、success/message；没有 transformer loading 输出。 |
| 7. 哪些进入 observation | LMP、MEF+、MEF-、归一化最小电压、最大线路 loading、网损、security penalty、OPF success，共 8 维。 |
| 8. 哪些进入 reward | 正式双 agent 配置中没有 OPF/LMP/MEF/电压/线路项直接进入 reward；`enable_grid_reward=false`。基础 reward 使用本地购电量、电价、外生 carbon factor 和购电峰值。 |
| 9. 哪些只记录 | OPF 发电成本/出力/排放、参考 USEP、电压/线路违约、LMP、MEF、损耗、cache 命中等主要用于 observation 和日志；其中很多不参与正式 reward。 |

关键证据：`envs/idc_price_env.py::step`、`env_wrappers/grid_coupled_env.py::_run_grid_update`、`grid_model/opf_solver.py::_solve_opf`。

## 3. 算力侧实现

### 3.1 Task 模型

`idc_model/task.py::Task` 真实具有 `arrival_time`、`duration`、`load_profile`、`workload`、`deadline`、`priority`、`interruptible`、`parallelizable`、`remaining_work`、`status`、`start_time`、`finish_time`、`assigned_servers` 和 `execution_log`。

| 属性 | 是否存在 | 是否真实参与当前正式环境 |
|---|---:|---|
| arrival time | 是 | 控制到达激活，也进入当前/全天任务到达观测。 |
| workload / remaining work | 是 | 是真实队列与执行扣减基础。 |
| duration / load profile | 是 | 主要用于生成 workload；执行时不是按固定 profile 逐小时绑定运行。 |
| deadline | 是 | 进入任务排序、紧急/逾期状态、deadline/SLA reward。 |
| priority | 是 | 进入排序、任务池观测和完成奖励。 |
| interruptible | 是 | 进入暂停判断、不可中断任务中断惩罚。 |
| parallelizable | 是 | 进入任务池观测；当前没有真实多 server 并行拆分机制。 |
| status | 是 | `not_arrived/waiting/running/paused/finished` 实际更新。 |
| pause/resume | 是 | 由环境动态维护 `is_paused`、计数和 reward；不是 Task dataclass 的显式字段。 |
| waiting time | 可计算 | 由时间戳计算，并进入等待压力/reward 和 info；不是持久字段。 |
| turnaround time | 可计算 | 环境 info 可输出平均 turnaround；标准训练 CSV 未将其列为 canonical episode metric。 |
| assigned_servers | 字段存在 | 当前正式调度不填充，没有真实任务—服务器绑定。 |

每个 episode 生成 30 个随机任务，并插入 1 个初始 backlog task，共 31 个 Task 对象。初始 backlog 的实际 workload 为 `Q0(300) * task_workload_scale(100) = 30000`。四类任务 profile 中，A 类不可中断/不可并行，B/C 类可中断/可并行，D 类可中断/不可并行。调度并非逐服务器指派，而是对任务排序后用聚合 capacity 顺序扣减 `remaining_work`。

任务排序分数由 urgent preference 和 continuity preference 加权；urgent 子项结合 deadline proximity、priority 和 overdue，continuity 子项偏向已启动/暂停任务。`parallelizable` 没有触发真实拆分，因此论文中不能写成“已实现可并行任务分配”。

### 3.2 服务器与 IDC 规模

当前不是 20 台独立物理服务器，而是 **20 个 server group**；每组代表 100 台等效服务器，共 2000 台等效服务器。配置证据为 `IDC_SCALE_CONFIG`：`enable_server_group_model=true`、`server_group_size=100`、`num_server_groups=20`。

异构性来自：

- 单机 idle 功率在约 160–240 W 随机采样，组功率乘以 100；
- 单机 max 功率在约 480–720 W 随机采样，组功率乘以 100；
- 单机计算能力围绕基准值做 ±25% 随机变化，再乘以组规模；
- 计算效率使用 `C_server / P_max`；
- 每组动作负载受 `max_task_load_per_server=0.60` 限制，并叠加 `base_load=0.05`。

没有 GPU/CPU 类型、服务器网络拓扑、机架、迁移通信、任务数据依赖或逐机开关机状态。

### 3.3 IDC 功耗公式

`idc_model/power_model.py` 使用：

```text
P_IT = sum_i [P_idle_i + (P_max_i - P_idle_i) * (2 L_i - L_i^1.3)] + 5000 W
COP = max(0.05 * (T_target - T_amb) - 0.1 * mean(L) + 3.0, 0.1)
P_cooling = P_IT / COP
PUE = 1 + 1/COP + P_others/P_IT
P_IDC = P_IT + P_cooling + P_others
P_others = 2000 W
```

组 idle/max 功率按 100 倍缩放，但固定 `delta_P_loss=5000 W` 与 `P_others=2000 W` 未随组规模缩放。温度观测仅是 `T_amb + 15 * previous_load` 的估计量，不是有热惯性的动态热模型。

## 4. 能源侧实现

### 4.1 PV

- 数据来源：`DATA_CONFIG` 没有外部 CSV，使用 `data_io/data_loader.py::_build_default_pv_curve` 的 6:00–18:00 正弦曲线，峰值 500 kW。
- forecast：有，完整 24h `pv_t` 每一步都进入 forecast observation；这里是 episode 已实现值，不是带误差的预测模型。
- agent/可控性：PV 不是 agent，也不可控。
- 功率平衡：`pv_used = min(pv_available, max(P_IDC + charge - discharge, 0))`。
- 弃光/上网：有 `pv_curtail`；`allow_pv_export=false`，不允许卖电。
- 指标：info 中有可用、使用、弃光、PV utilization rate 和相对 IDC 能耗的比例；标准 logger 持久化可用量/使用量，未把所有比例均列为 canonical 列。

### 4.2 BESS

正式缩放后参数：容量 10000 kWh，最大充/放电功率均为 2000 kW，初始 SOC 0.50，范围 `[0.10,0.90]`，目标 SOC 0.50，episode 末容差 0.05，充/放电效率均为 0.95，吞吐退化成本 0.02/ kWh。

能源 agent 实际只控制 **一个连续标量**。actor 输出 `[0,1]`，环境映射为 `[-1,1]`：负数充电，正数放电，0 空闲。因使用同一有符号变量，充放电天然互斥。环境按功率限制、SOC 可用能量和不得超过 IDC 需求进行硬裁剪；期望功率与实际功率差形成 soft invalid-action penalty。SOC 上下限为硬约束；最终 SOC 不是硬等式，仅在超出目标±0.05时给 episode 末惩罚。

退化模型只是充放电 throughput 的线性成本，没有循环深度、温度、日历老化或寿命状态。

## 5. 电网侧实现

### 5.1 网络和耦合点

- 使用 `pandapower.networks.case14()`：14 bus、15 line、5 transformer，保留原生基础负荷、发电机与 external grid。
- 仅对发电机电压设定与 bus OPF 上下限做一致化，不改拓扑。
- 单 IDC、单聚合 BESS、单聚合 PV、单 PCC；均等效接 IEEE bus 9（pandapower index 8）。
- 不存在多 IDC、多微电网或多个并网点。
- NEMS 24h `grid_load_scale` 文件真实存在，并按小时同时缩放 IEEE-14 原生有功/无功负荷；历史正式运行元数据记录“Loaded dynamic grid load scale”。同文件 USEP 仅作参考日志。

### 5.2 AC OPF

AC OPF 是真实执行，不是接口占位：`grid_model/opf_solver.py` 深拷贝网络、缩放基础负荷、在 bus 8 增加 IDC 的 `p_mw` 且 `q_mvar=0`，随后调用 `pandapower.runopp`。每个环境 step 都请求 OPF，但 cache hit 时复用结果；reset 还执行零 IDC 负荷的初始化 OPF。

输出包括：ext_grid+gen 有功出力、总发电成本、bus `lam_p`、bus voltage、line loading、总负荷、总发电、network loss 和 success/message。没有 transformer loading 的提取与日志指标，也没有 IDC/PV/BESS 无功或逆变器控制。

OPF 失败返回 `success=false` 和可用的部分/空结果。正式双 agent 环境不会因此终止 episode；非有限网架观测被转为 0，success flag 为 0。由于 grid reward 关闭，失败本身不改变正式 reward，只进入观测、safe-cost 日志和成功率统计。

### 5.3 MEF 与 cache

MEF 使用基准、`+0.1 MW`、`-0.1 MW` AC OPF 的有限差分。发电排放因子是代码内确定性默认假设，不是外部实测数据；因此 MEF 是“由真实 OPF dispatch + 假设排放因子计算的近似边际量”。

每个 worker 拥有独立 LRU cache。OPF key 包含 mode、IDC bus、hour、按 0.005 分箱的 grid load scale、按 0.1 MW 分箱的 IDC load；MEF 使用 0.01 MW load bin 和 delta。最大 50000 条，不在 reset 时清空，不缓存失败结果。cache 提高速度但引入分箱近似。

### 5.4 电网指标事实分类

| 指标 | 状态 | 说明 |
|---|---|---|
| bus voltage | 真实计算 | AC OPF `res_bus.vm_pu`；观测只保留聚合最小值。 |
| line loading | 真实计算 | `res_line.loading_percent`；观测/日志主要保留最大值。 |
| transformer loading | 未形成输出 | 变压器存在于网络和 HGTA 拓扑，但 solver/logger 没有提取 loading。 |
| generation | 真实计算 | ext_grid 和 gen 有功出力；主要用于 OPF 排放/MEF和 info。 |
| local grid import | 本地计算 | `P_grid=max(P_bus_net,0)`，不是专门提取的 OPF ext-grid import 指标。 |
| LMP | 求解器结果 | 使用 `res_bus.lam_p`；不是外部市场价格。 |
| MEF | 近似计算 | AC OPF 有限差分 + 默认发电排放因子。 |
| carbon factor | 外生/合成 | 基础 reward 使用当前外生 `carbon_factor_t`，与 OPF MEF/总排放是两套量。 |
| OPF success | 真实状态 | 每步记录。 |
| voltage/line violation | 真实结果派生 | 由 OPF voltage/line loading 与阈值比较。 |

## 6. 双智能体、动作和观测

### 6.1 IDC Agent

- 原始 local observation：完整的 288 维 wrapped observation。
- actor 实际输入：288 维。
- 有效动作：22 维，包括 20 个 server-group 任务执行强度、1 个 urgent preference、1 个 continuity preference。
- 物理含义：server 动作乘 0.60 得到计划任务负载；两个 preference 影响任务排序，不直接分配具体服务器。
- policy 输出：22 维 bounded diagonal Gaussian 动作，环境接收 `[0,1]`。

IDC observation 的 288 维为：6 当前全局量 + 10 task-pool 量 + 120 server-group 量 + 144 全天前瞻量 + 8 网架量。

### 6.2 BESS Agent

- 真实 local observation：164 维，即当前全局前 6 维 + 全天 forecast 144 维 + grid 8 维 + supplemental 6 维。
- supplemental：SOC、BESS energy、上一转移的 `P_IDC`、`P_grid`、charge、discharge。
- actor 兼容输入：164 维后补 124 个零，对齐为 288 维。
- 有效动作：1 维 BESS charge/discharge 标量。
- 对齐动作：22 维，其中第 0 维有效，第 1–21 维为 dummy。

### 6.3 padding 是否影响训练

`marl/specs/agent_specs.py` 定义 IDC/BESS padded action 均为 22，有效维分别为 22/1。`HarlPaddedBridge` 只把 BESS 第 0 维传给物理环境；`effective_action_mask.py` 在 log-ratio、clip、KL、entropy 和 HAPPO factor 中屏蔽 21 个虚拟维度。诊断还检查 dummy mean rows 与 log-std 的梯度为 0。

结论：dummy action 仍被网络输出和采样，增加参数/存储/采样开销，但按当前 loss 实现不参与 PPO/HAPPO ratio、entropy 或 factor，也不应给虚拟输出行产生梯度。共享 trunk 仍由真实第 0 维训练。

### 6.4 centralized state 与信息边界

centralized state 为 294 维：完整 wrapped observation 288 + supplemental 6。两个 actor observation 不是简单拼接；IDC 和 BESS 分别由 builder 构造。MLP/HGTA critic 均使用同一个 centralized state，并输出共享 `V(s)`。

信息风险：IDC actor 能看到完整任务池、全部 20 组服务器、网架聚合信息和完整 24h price/temperature/task-arrival/PV 序列。特别是 `lambda_t` 来自 episode reset 后已生成的真实任务到达工作量，而非带误差预测，因此属于强全知前瞻假设。BESS actor 不看 task-pool/server 细节，但也看完整 24h 前瞻。carbon factor 不在 actor observation 的当前/forecast特征中，尽管 reward 使用它。

## 7. MAPPO + MLP

actor 使用 HARL `StochasticPolicy`：feature normalization、两层 `[32,32]` ReLU MLP、orthogonal initialization、bounded diagonal Gaussian 输出；两个 agent 各有独立 actor，不共享参数。BESS actor 因 padding 也采用 288 输入、22 输出的同规模网络。

critic 使用 HARL `VNet`：294 维 centralized state，feature normalization，两层 `[32,32]` MLP，输出单个 `V(s)`。它不是每 agent 一个 value，也不是 Q-function。

EP rollout buffer 存 shared state、local observation、动作、log-prob、共同 reward、mask 和 value。GAE 使用共享 reward 和 centralized `V(s)`，`gamma=0.99`、`lambda=0.95`。PPO 使用 clip 0.2、1 actor epoch、1 critic epoch、各 1 mini-batch、entropy coefficient 0.01、clipped value loss 和 Huber loss。

MAPPO update 对两个独立 actor 依次调用 PPO train，但不使用 HAPPO factor；其语义在审计元数据中标为 `parallel_independent`，即第二个 actor 的 loss 不因第一个 actor 已更新而修正。

**精确描述：当前 MAPPO+MLP 是两个独立前馈高斯 actor + 一个共享 294 维 centralized MLP value critic 的 CTDE PPO；共享 reward/GAE，不共享 actor 参数。**

## 8. HAPPO + MLP

HAPPO 与 MAPPO 的真实差异位于 `marl/algorithms/update_strategy.py::HAPPOUpdateStrategy`：

1. factor 初始化为 1；
2. 按配置顺序逐 agent 更新，正式配置固定为 IDC 后 BESS；
3. 每个 agent 更新前将当前 factor 写入 buffer；
4. 更新前后重新计算该 agent 的有效动作 log-prob；
5. 计算有效维 importance ratio，并乘入 factor；
6. 后续 agent 的 HAPPO actor loss 使用更新后的 factor；
7. 两个 actor 完成后更新共享 critic。

因此这不是“只把 PPO 改成顺序更新”。它包含 HAPPO 的顺序更新与乘法 importance-factor correction。advantage 仍由共享 centralized value 和 shared reward 的 GAE 得到；没有额外独立的 agent-specific advantage estimator。

分类：**B. 基本遵循 HAPPO 核心机制，但有工程适配/简化。**

主要简化：仅两个 agent；固定更新顺序；1 PPO epoch/1 mini-batch；前馈网络；共享单一 team advantage；异构动作通过等维 padding + effective mask 适配；没有在当前仓库重新实现完整官方训练框架，actor/critic 基类来自固定 HEAD 的同级 HARL。

## 9. HGTA 实现

### 9.1 使用位置

HGTA 只用于 centralized critic。IDC actor 和 BESS actor 与 HAPPO+MLP 完全相同，仍是 MLP。`HAPPO_HGTA` 不是 graph actor，也不是 actor-critic 双图网络。

### 9.2 节点与特征

图固定为 39 个节点：

| node type | 数量 | raw feature |
|---|---:|---|
| global | 1 | 当前 temperature、price、task arrival、time sin/cos；grid LMP、MEF+、MEF-、min voltage、max line loading、network loss、security penalty、OPF success，共 13 维。注意没有当前 backlog Q。 |
| idc | 1 | `P_IDC`、`P_grid`，2 维。 |
| task_pool | 1 | waiting/running/finished/unfinished count、urgent/overdue work、平均剩余 deadline、平均 priority、parallelizable/interruptible work，共 10 维。没有单任务节点。 |
| server_group | 20 | previous load、capacity、efficiency、unit cost、available、estimated temperature，每节点 6 维。 |
| bess | 1 | SOC、energy fraction、charge fraction、discharge fraction，4 维。 |
| pv | 1 | 当前小时 forecast、全天 forecast mean、全天 forecast max，3 维。 |
| bus | 14 | nominal voltage、基础有功/无功负荷、has generator、has ext-grid、has IDC、归一化 bus index，7 维。均为静态模板。 |

每种 node type 使用独立 `Linear(raw_dim,32)` 投影，没有 raw-feature padding/mask；随后均为 32 维 hidden embedding。

### 9.3 边

14 种 relation，共 90 条有向 relation instance：

- 20 条 `server_group -> idc belongs_to`；
- 20 条 `idc -> server_group contains`；
- `task_pool <-> idc` 各 1；
- `idc <-> bus[8]` 各 1；
- `bess <-> bus[8]` 各 1；
- `pv <-> bus[8]` 各 1；
- `global <-> idc` 各 1；
- IEEE-14 15 条 line 各转为双向边，共 30；
- IEEE-14 5 个 transformer 各转为双向边，共 10。

line 与 transformer 是不同 relation type，拓扑真实来自 pandapower IEEE-14。没有阻抗、容量、潮流等 edge feature，边只携带连接关系和 relation identity。global node 只与 IDC 双向连接，并没有连到所有节点。

### 9.4 Attention 和维度链

```text
294-d centralized state
  -> GraphBuilder 切片为 7 类、39 个 raw nodes + 144-d forecast
  -> 7 个 type-specific input Linear -> 每节点 32-d
  -> 2 层 heterogeneous relation attention
       每类独立 Q/K/V/output projection 和 LayerNorm
       4 heads × 8-d/head
       每 relation 独立 4×8×8 key transform
       每 relation 独立 4×8×8 value transform
       每 relation 独立 head bias
       对同一 target node 的所有 incoming relation 联合 softmax
       edge-index message passing + residual + LayerNorm + ReLU
  -> singleton type identity readout；server/bus type mean pooling
  -> 7 type summaries × 32 = 224-d graph representation
  -> 独立 forecast MLP：144 -> 64 -> 32
  -> concat：224 + 32 = 256
  -> value head：256 -> 64 -> 1
  -> 单个 shared V(s)
```

它真实使用 multi-head Q/K/V relation attention 和 edge-index message passing，不是装饰性图接口。它有 residual、LayerNorm、activation、dropout 配置；但没有标准 Transformer block 中独立的两层 position-wise FFN 子层，只有 attention 后 output linear，以及最终 forecast/value MLP。因此准确说法是“纯 PyTorch 异构关系注意力图编码器”，不应无条件等同于完整标准 Graph Transformer block。

### 9.5 异构性来源

- A 不同 node type 使用不同 encoder：**是**。
- B 不同 edge/relation 使用不同 attention 参数：**是**。
- C 显式 node-type embedding：**否**；类型信息由独立模块参数隐式编码。
- D 显式 relation embedding vector：**否**；relation 由独立 K/V 变换和 bias 参数化。
- E 不同类型消息分别构造并按目标节点联合聚合：**是**。
- F 仅因输入维度不同后统一投影：**否**，后续 Q/K/V/output 也按 node type 独立，relation 参数也独立。

### 9.6 global node 与 value

global node 参与两层 message passing，但只与 IDC 双向连接。最终 value 不直接从 global node 读取，而是对七种类型分别 identity/mean readout 后拼接。需要注意，完整 144 维全天 forecast 绕过图结构，经独立 MLP 直接拼到 graph representation，因此 critic 可以不依赖拓扑路径使用强前瞻信息。

另一个关键事实是：14 个 bus node 的 feature 全为静态模板；动态 LMP、MEF、电压、线路 loading 和损耗被放在 global node，而不是对应 bus/line 上。因此图确实传播 IEEE 拓扑，但没有逐 bus 动态 voltage/load 或逐 line 动态 flow/loading。电网拓扑在当前 critic 中主要编码静态结构，而非动态空间潮流场。

## 10. HAPPO+MLP 与 HAPPO+HGTA 公平性

当前配置中两者只改变 critic factory 的 `critic.type`。两个 actor、actor optimizer、actor LR、rollout、reward、observation、动作 mask、gamma、GAE、clip、batch、epochs、entropy、value loss、worker seeds 和更新顺序相同。critic LR 同为 0.0005。critic 初始化使用 `base_seed+200000=207110` 的隔离 RNG，避免不同 critic 构建消耗 actor sampling RNG。

在当前配置/已保存模型上统计的 trainable parameters：

| 模块/方法 | 参数量 |
|---|---:|
| IDC actor | 11,756 |
| BESS actor（含 21 个 dummy 输出） | 11,756 |
| MLP critic | 11,245 |
| HGTA critic | 102,673 |
| HAPPO+MLP 总计 | 34,757 |
| HAPPO+HGTA 总计 | 126,185 |

HGTA critic 约为 MLP critic 的 9.13 倍，总参数量约为 3.63 倍。故“只改变 critic 类型”在代码控制变量层面成立，但不是参数量匹配的公平比较。hidden dimension 名义上都为 32，也不代表容量相同。

## 11. Reward 结构

正式环境对两个 agent 复制同一个标量 reward，是 **shared team reward**，没有 agent-specific reward。

正式缩放后的主要 reference：`queue_ref=600000`、`queue_capacity_ref=600000`、`cost_ref=6000`、`carbon_ref=1500`、`sla_ref=5000`、`grid_power_limit=1800 kW`、`peak_power_ref=1000 kW`。任务总数通常为 31。

| 项 | 公式含义与 normalization | weight | 级别 |
|---|---|---:|---|
| completed work | `+ completed_work/queue_ref` | 5.0 | step |
| finished tasks | `+ newly_finished/31` | 1.5 | step |
| priority finish | `+ newly_finished_priority/(5*31)` | 0.6 | step |
| electricity cost | `- cost_t/cost_ref` | 0.35 | step |
| carbon | `- P_grid*dt*external_carbon_factor/carbon_ref` | 0.30 | step |
| backlog | `- Q_next/queue_ref` | 0.8 | step |
| queue overflow | `- max(Q_next-capacity_ref,0)/queue_ref` | 1.2 | step |
| urgent backlog | `- urgent_work/queue_ref` | 0.8 | step |
| waiting | `- average_waiting_pressure/horizon` | 0.25 | step |
| deadline miss | `- newly_missed/31` | 1.2 | step |
| SLA | `- priority-weighted overdue penalty/sla_ref` | 0.8 | step |
| unused capacity | `- unused_capacity/queue_ref` | 0.08 | step |
| grid peak | `- max(P_grid-limit,0)/peak_ref` | 1.0 | step |
| pause | `- pauses/31` | 0.15 | step |
| resume | `- resumes/31` | 0.03 | step |
| noninterruptible interruption | `- count/31` | 0.8 | step |
| load smooth | `- mean(abs(load_t-load_{t-1}))` | 0.05 | step |
| action smooth | `- mean(abs(action_t-action_{t-1}))` | 0.03 | step |
| BESS degradation | `- throughput_cost/cost_ref` | 1.0 | step |
| invalid BESS request | `- clipped_power_difference/max_bess_power` | 0.2 | step |
| final backlog | `- Q_final/queue_ref` | 3.0 | episode end |
| final SOC | `- max(abs(SOC-target)-0.05,0)` | 2.0 | episode end |

电网电压、线路 loading、OPF failure、LMP、MEF 的 grid-reward 权重均为 0，正式配置关闭 grid reward。reward 因而高度工程化且包含 22 个 shaping 项；论文指标若主打电网安全或 OPF 排放，必须注意它们与当前优化目标并不一致。

## 12. 约束与 Safe RL

### 当前正式双 agent

- 没有 Lagrangian PPO、可学习 lambda、cost critic 或 CMDP actor-critic。
- `safe_rl_enabled=false`。
- 硬约束：动作范围、BESS 功率、SOC 上下限、充放电互斥、不得放电超过本地需求、无卖电。
- soft penalty：队列/超时/SLA/峰值/BESS invalid/final SOC 等 reward 项。
- 电压、线路和 OPF failure 被计算为 `safe_cost`/violation 日志，但不用于拉格朗日优化，也不直接进入正式 reward。

### 历史单 agent

`safe_rl/` 中存在基于单 agent SB3 PPO 的外部 Lagrangian/constraint wrapper，可维护电压和 OPF violation multiplier，部分模式可在 OPF failure 时终止。它不属于当前正式双 agent MAPPO/HAPPO/HGTA 链，不能在论文中混称为当前方法的 constrained MARL。

## 13. 当前可直接输出的评价指标

### 13.1 计算服务

- completed work、workload completion rate；
- finished task count、task completion rate；
- backlog/queue/overflow/urgent backlog；
- deadline miss step/total，deadline miss rate 可由 info 或计数计算；
- SLA violation count 与 SLA penalty；
- average waiting time；
- average turnaround time 在环境 info 中存在，但不属于标准训练 CSV 的 canonical episode 列；
- pause、resume、noninterruptible interruption；
- server load mean/max、available capacity、PUE。

当前没有持久化 P50/P95/P99 waiting/turnaround、按任务类型分组 SLA、JCT 分布或公平性指标。

### 13.2 能源与碳

- IDC/IT/cooling/grid energy；
- step/total electricity cost；
- `P_IDC`、`P_grid`、episode peak、peak excess；
- PV available/used/curtailed、利用率（info）；
- BESS physical action、charge/discharge power 与 energy、SOC、throughput、degradation、invalid request；
- step/total carbon、carbon factor、carbon cost（但正式 `carbon_price=0`）。

没有直接的 carbon per completed work/task canonical metric，可由已有总量离线计算；没有生命周期 BESS carbon、需量电费或售电收益。

### 13.3 电网

- OPF/MEF success rate；
- min/max/mean voltage 聚合；
- max line loading 与 line violation；
- network loss；
- LMP、MEF+、MEF-；
- grid generation cost/emission 和 generator dispatch 可在 info 获取；
- OPF/MEF cache hit、miss、size、hit rate；
- safe violation/cost 日志。

没有 transformer loading、逐 bus/逐 line canonical 时序、N-1 security、频率、备用、潮流方向或 reactive power quality 指标。

## 14. 训练与实验基础设施

| 能力 | 当前状态 |
|---|---|
| random seed | 支持。base seed 7110；两个 worker 使用 7110/8110；task/server/action space 均显式播种。 |
| 多 seed | 单次运行一个 seed，可通过多次 CLI 运行实现；没有内置统一 multi-seed sweep/聚合器。fixed suite 支持多个 scenario seed。 |
| deterministic eval | 支持，单 worker，actor deterministic action。 |
| fixed scenario eval | 支持；suite 持久化任务、外部序列、环境状态并保存 hash。 |
| checkpoint | 支持周期和 final checkpoint。 |
| exact resume | 支持 post-update 边界恢复；仓库有连续/恢复对照产物。不是任意 mid-rollout resume。 |
| optimizer state | 两个 actor optimizer 和 critic optimizer 均保存/恢复。 |
| RNG/environment state | Python、NumPy、Torch/CUDA RNG、每 worker 环境、Task、server/task RNG、grid cache、logger 等均恢复。 |
| scheduler state | 不支持/不适用。正式配置关闭 LR decay；入口在 exact-resume 模式下拒绝 linear LR decay。 |
| training step resume | 只支持完整 update 后边界；不支持 episode/rollout 中间任意 step。 |
| CSV | step、episode、update CSV 均支持。 |
| TensorBoard | 支持。 |
| evaluation logging | CSV、JSON manifest、aggregate metrics。 |
| best model | logger 只记录 best episode 元数据，不保存独立 best weights。 |
| final model | 始终保存两个 actor、critic 和 final checkpoint。 |

checkpoint 是真正的连续训练恢复，而不只是载入权重；但精确性边界严格限定在 post-update，恢复后会创建新的 run segment/日志目录，而非原 CSV 原地续写。

环境依赖由 `environment_harl.yml` 声明：Python 3.10、PyTorch、Gymnasium 1.2.3、NumPy 2.2.6、pandapower 3.4.0 等。当前可用环境为 `C:\Users\bulio\miniconda3\envs\idc_ppo`。正式 MARL 源码依赖同级 HARL 路径；直接运行部分测试若未设置 `PYTHONPATH=C:\Users\bulio\Desktop\IDC\HARL` 会报 `ModuleNotFoundError: harl`。

## 15. 工程正确性验证证据

### 15.1 本次审计实际运行

在当前工作树、`idc_ppo` Python 和同级 HARL 路径下，本次执行：

```text
test_hgta_graph_builder.py
test_hgta_encoder.py
test_hgta_critic_interface.py
test_unified_algorithm_interface.py
```

结果：**26 passed in 16.01s**。这直接验证当前图构建、relation attention、HGTA backward/critic 接口和三方法统一接口。

动作 adapter/effective mask/padding 组合测试共收集 16 项；运行显示前 13 项通过后，在首个完整物理 episode 回归项处超过 120 秒而被审计限时终止，因此不能写成 16/16 当前通过。前 13 项覆盖动作 round-trip、mask、ratio、entropy、KL、虚拟输出梯度等单元逻辑；后续环境级 padding parity 本次未完成。

checkpoint resume 也因与上述长组同跑而被 120 秒限时终止，故本次没有新的完整 resume pytest 结论。

### 15.2 仓库内可确认的历史成功产物

- 三种方法均有 1-update 正式 run、final model 和 final checkpoint：`runs/part18_rng_fix_*_1update*`。
- MAPPO+MLP、HAPPO+MLP、HAPPO+HGTA 均有 5-update gate 目录。
- HAPPO+HGTA 有从 update 3 恢复至 update 5 的 run，metadata 标记 `resumed=true`、`post_update_only` 并存在 final checkpoint。
- `evaluations/part18_rng_fix_three_methods_actor_only/evaluation_manifest.json` 状态为 `completed`、`failure_count=0`，三个 final checkpoint 在固定 scenario 上完成 deterministic actor-only evaluation；manifest 同时确认无 actor gradient、actor 参数未变化、critic metric 不参与评估。

这些产物证明“短程训练、保存、载入、固定评估曾成功执行”，但不证明当前 dirty 工作树与产物源码逐字一致，也不构成正式多 seed 性能验证。

### 15.3 存在但不能仅凭脚本宣布最新通过的测试

仓库还包含 env reset/step、双 agent shape/rollout/parity、padding invariance、并行采样、formal entry、5-update gate、RNG isolation、checkpoint resume、fixed evaluation、metrics logger、重复运行精确性和三方法公平审计脚本。除上述当前测试和明确历史 artifact 外，其余应表述为“存在测试脚本/历史报告，最近一次在当前工作树上的完整通过状态未在本次确认”。

## 16. 当前系统主要简化假设

| 简化事实 | 性质 |
|---|---|
| 单 IDC、单聚合 PV、单聚合 BESS、单 PCC | 合理首版抽象；也是拓扑复杂度和泛化范围的明显限制。 |
| 20 个 server group 代表 2000 台等效服务器 | 可计算的聚合抽象；丢失逐服务器/机架/GPU 拓扑与通信。 |
| task pool 在 HGTA 中只有 1 个聚合节点 | 控制图规模；丢失单任务关系和任务—服务器分配图。 |
| 24h horizon、1h time step | 日前调度抽象；无法描述分钟级 BESS/热/计算波动。 |
| PV 不可控、不卖电 | 合理初版；限制微电网交易与逆变器控制研究。 |
| BESS 单聚合标量动作 | 清晰但动作空间极小；无多储能、无 reactive power。 |
| 线性 throughput 退化 | 简化；没有 DoD、温度、循环寿命或日历老化。 |
| 无热惯性 | `T_amb+15*load` 仅观测估计；制冷 COP 为瞬时函数。 |
| 已知完整 24h price/temperature/PV/task arrivals | 强确定性 forecast 假设；任务 arrival 是已生成的真实 future。 |
| carbon factor 未进入 actor obs | reward 使用当前 carbon factor，但 actor 没有显式看到它。 |
| AC OPF 每小时准静态 | 无 ramp、unit commitment、频率、动态稳定性、N-1。 |
| IDC/PV/BESS 仅作为 net active load | `q=0`，没有逆变器、电压/无功控制或独立设备 OPF 变量。 |
| bus 动态状态聚合进 global node | HGTA bus node 仍是静态；不建模逐 bus 动态空间场。 |
| 默认合成 price/carbon/temp/PV | 除 NEMS load scale 外，正式环境没有外部实测时序。 |
| 共享 team reward | 便于协同；不显式表达个体收益、博弈或独立约束。 |

## 17. 潜在论文风险点（基于代码事实）

| 风险问题 | 审计判断 |
|---|---|
| HGTA 只是 MLP 前图编码 | 部分成立：graph encoder 后确有 value MLP，但图编码本身有真实 relation attention/message passing。更准确是“图表示 + 独立 forecast MLP + value MLP”。 |
| 图边未真正使用 | 不成立。edge index 真实参与逐 relation attention 和消息聚合。 |
| 电网拓扑是静态装饰 | 部分成立且重要。拓扑真实使用，但 bus feature 静态，动态网架量集中在 global，缺少逐 bus/line 动态状态与 edge electrical feature。 |
| relation 共享同一参数 | 不成立。每个 relation 有独立 K/V transform 和 bias。 |
| global node 过强绕开图 | global 只连 IDC，不是全连接超级节点；但 144 维 forecast 通过独立 MLP 直接绕开图，是更明显的 shortcut。 |
| actor 看全部全局状态，CTDE 变弱 | IDC actor 确实看完整 raw wrapped obs；BESS local obs 做了裁剪。CTDE 并非失效，但 IDC 侧 decentralized information boundary 较弱。 |
| HGTA 参数远大于 MLP | 成立：critic 约 9.13 倍，总参数约 3.63 倍。 |
| reward 人工规则过多 | 成立：22 个正负 shaping/terminal 项。 |
| reward 与论文指标不一致 | 部分成立。计算、成本、碳、峰值一致；OPF emission、LMP/MEF、电压和线路安全未进入正式 reward。 |
| 两 agent 没有明显耦合 | 不成立。IDC 负荷改变成本/碳/峰值/PV消纳/OPF/BESS机会，BESS 改变共享购电与网架负荷。但共享 reward、1 维 BESS 动作和动作维度不对称可能让冲突较弱，需要实证。 |
| BESS 影响太弱 | 代码上上限 2 MW、容量 10 MWh，影响真实存在；相对典型 IDC 功率和 IEEE-14 基础系统的敏感度尚未系统量化。 |
| MAPPO/HAPPO 差异不足 | 代码差异真实存在：HAPPO factor correction；但仅 2 agent、固定顺序、1 epoch，性能差异可能小，需消融。 |
| HAPPO 非标准实现 | 应表述为分类 B：遵循核心 factor/sequential update，有工程简化；不宜称完整无简化官方复现。 |
| graph state 含未来真实信息 | 成立，完整 `lambda_t`、price、temperature、PV 全日序列进入 state/actor。 |
| PV/price/carbon forecast 假设过强 | PV/price/task/temp 为全天完全已知；carbon 不在 forecast observation，但当前 reward 使用当前 carbon。 |
| 单 IDC 使图简单 | 成立：只有一个 IDC/task pool/BESS/PV，跨 IDC 或跨微电网 relation 未实现。 |
| 电网信息未显著影响决策 | 直接 reward 影响缺失，bus 动态空间信息也缺失；但 8 维 grid obs 可间接影响 actor/critic。是否“显著”必须通过去 grid obs、去 topology、扰动负荷等实验确认。 |
| 当前代码难以提交复现 | 成立：大量正式文件 untracked，且依赖同级 HARL 固定 HEAD。正式实验前必须形成可追溯 commit/环境锁定。 |

## 18. 论文事实基线表

| 内容 | 当前状态 | 主要证据文件/函数 | 验证状态 |
|---|---|---|---|
| 双 agent 环境 | 已实现 | `marl/envs/idc_grid_multi_agent_env.py` | 历史 run + 接口测试存在 |
| IDC actor 22 有效动作 | 已实现 | `envs/idc_price_env.py::step`、`agent_specs.py` | 已有单元/历史产物 |
| BESS actor 1 有效动作 | 已实现 | 同上 | 已有单元/历史产物 |
| observation padding | 已实现 | `marl/bridges/harl_padded_bridge.py` | 单元逻辑本次部分通过；完整环境回归本次超时 |
| effective action mask | 已实现 | `marl/algorithms/effective_action_mask.py` | 本次前置单元项通过；历史诊断显示虚拟梯度为 0 |
| MAPPO+MLP | 已实现 | `method_registry.py`、`update_strategy.py` | 有 1/5-update 历史产物 |
| HAPPO+MLP | 已实现 | `HAPPOUpdateStrategy` | 有 1/5-update 历史产物 |
| HAPPO factor correction | 已实现 | `update_strategy.py` | 当前统一接口测试通过；历史 audit 指标存在 |
| HAPPO+HGTA | 已实现 | `critic_factory.py`、`hgta_critic.py` | 当前 HGTA/统一接口 26/26；历史 1/5-update |
| HGTA 仅用于 critic | 已实现事实 | `hgta_critic.py`、runner | 代码确认 |
| 39-node graph | 已实现 | `graph_schema.py`、`graph_builder.py` | 当前 graph tests 通过 |
| IEEE topology edges | 已实现 | `graph_schema.py::build_graph_schema` | 当前 graph tests 通过 |
| relation-specific attention | 已实现 | `hgta_encoder.py::HeterogeneousRelationAttentionLayer` | 当前 encoder tests 通过 |
| type-specific encoder/QKV | 已实现 | `hgta_encoder.py` | 当前 encoder tests 通过 |
| 显式 node/relation embedding | 未实现 | — | — |
| 动态逐 bus/line graph feature | 未实现 | bus feature 为静态模板 | — |
| IEEE-14 AC OPF | 已实现 | `ieee14_loader.py`、`opf_solver.py` | 历史正式 evaluation OPF success=1.0；本次未重跑完整训练 |
| 动态 NEMS grid load | 已实现 | `grid_coupled_env.py`、CSV | 文件存在且历史 metadata 确认加载 |
| OPF/MEF cache | 已实现 | `grid_model/grid_cache.py` | 历史指标/测试脚本 |
| LMP | 已实现 | `res_bus.lam_p` | 历史 evaluation 有值 |
| MEF | 近似实现 | `mef_calculator.py` | 历史 success 指标；依赖假设排放因子 |
| transformer loading | 未实现输出 | — | — |
| PV | 已实现、不可控 | `data_loader.py`、`IDCPriceEnv20D` | 历史指标 |
| BESS SOC/效率/退化 | 已实现 | `IDCPriceEnv20D::step` | 历史指标 |
| 最终 SOC 硬约束 | 未实现；仅 soft penalty | `r_soc_final` | 代码确认 |
| 多 IDC | 未实现 | — | — |
| 多微电网/PCC | 未实现 | — | — |
| shared team reward | 已实现 | `IDCGridMultiAgentEnv.step` | logger 检查 shared diff |
| Lagrangian/Safe MARL | 未实现于正式双 agent | metadata `safe_rl_enabled=false` | 代码确认 |
| checkpoint 完整状态 | 已实现 | `training_checkpoint.py` | 历史 resume 产物 |
| 任意 mid-rollout resume | 未实现 | post-update only | — |
| deterministic fixed eval | 已实现 | `eval_harl_mappo_fixed.py` | 历史 completed manifest，failure=0 |
| best model weights | 未实现 | 仅 logger best metadata | — |
| final model | 已实现 | formal training entry | 历史文件存在 |
| 多 seed 正式统计实验 | 尚未完成 | 当前仅 smoke/gate | 未验证 |

## 19. 最终四个结论

### 结论 1：当前整个系统实际上是什么（不超过 300 字）

当前系统是一个 24 小时、1 小时步长的单 IDC—单聚合 PV/BESS—单 PCC IEEE-14 协同控制环境。IDC agent 控制 20 个服务器组的执行强度与两项任务排序偏好，BESS agent 控制一个充放电标量；任务执行形成 IDC 功耗，经 PV/BESS 平衡后作为 bus 9 有功负荷进入逐时 AC OPF。两 agent 使用共享 reward，支持 MAPPO/HAPPO、集中式 critic、固定评估和完整 post-update checkpoint 恢复。

### 结论 2：HAPPO+HGTA 相对 HAPPO+MLP 改变了什么（不超过 300 字）

它只替换集中式 value critic：HAPPO+MLP 将 294 维 centralized state 输入两层 MLP；HAPPO+HGTA 将同一 state 构造成 39 节点、14 relation 的异构图，经两层 type/relation-specific 多头注意力、类型级 pooling 和独立全天 forecast MLP 输出同一个 `V(s)`。两个 actor、HAPPO 顺序更新、factor、rollout 和超参数不变；HGTA critic 参数约为 MLP critic 的 9.13 倍。

### 结论 3：最可能成为论文核心的代码事实

在不评价创新性的前提下，当前最核心的方法差异事实是：集中式 critic 将 IDC 任务池、20 个异构服务器组、BESS、PV、IEEE-14 buses 和 global context 建模为类型/关系显式区分的图；节点类型有独立 encoder/QKV，关系有独立 K/V 变换与 bias，真实 IEEE line/transformer 拓扑参与两层多头消息传递，最终估计共享 `V(s)`。同时必须如实披露 bus 动态特征缺失、forecast MLP shortcut 和参数量不匹配。

### 结论 4：正式论文实验前必须确认的技术事实

1. 固化当前 dirty/untracked 工作树为可追溯 commit，并锁定 HARL HEAD 与环境。
2. 用当前最终 commit 重跑三方法多 seed 长训练、exact resume 和 fixed evaluation，而非继续依赖 1/5-update smoke。
3. 做参数量/容量匹配和 wall-clock 比较，分离“图结构收益”与“模型更大”。
4. 做 HGTA 消融：去 topology、relation sharing、静态 bus、去 global、去 forecast shortcut、MLP 同参数量。
5. 量化 grid obs/reward、BESS 动作和 AC OPF 对策略的真实敏感度；明确电网安全为何不在 reward。
6. 确定完整未来 task arrival/price/PV/temp 是完美预测实验条件还是信息泄漏，并增加 forecast error/rolling forecast 对照。
7. 决定论文范围是否接受单 IDC/单 PCC；若声称跨 IDC/多微电网协同，当前代码不支持。
8. 预先锁定论文主指标，并补齐 turnaround 分位数、carbon intensity、transformer/逐节点电网指标等缺口。

## 20. 关键证据索引

- 环境与 reward：`envs/idc_price_env.py`
- Task：`idc_model/task.py`、`idc_model/task_model.py`
- IDC 功耗：`idc_model/power_model.py`
- 正式配置：`configs/config_ultimate.py`、`configs/harl_mappo_short.yaml`
- PV 数据：`data_io/data_loader.py`
- 网架 wrapper：`env_wrappers/grid_coupled_env.py`
- IEEE-14/OPF/MEF/cache：`grid_model/ieee14_loader.py`、`opf_solver.py`、`mef_calculator.py`、`grid_cache.py`
- 双 agent：`marl/envs/idc_grid_multi_agent_env.py`
- observation：`marl/observations/`
- padding/mask：`marl/bridges/harl_padded_bridge.py`、`marl/specs/agent_specs.py`、`marl/algorithms/effective_action_mask.py`
- MAPPO/HAPPO：`marl/algorithms/update_strategy.py`、`marl/methods/method_registry.py`
- HGTA：`marl/graphs/graph_schema.py`、`graph_builder.py`、`marl/critics/hgta_encoder.py`、`hgta_critic.py`
- checkpoint：`marl/checkpointing/training_checkpoint.py`
- logging/metrics：`marl/logging/training_metrics.py`
- 固定评估：`eval/eval_harl_mappo_fixed.py`、`marl/evaluation/`

---

本报告没有修改任何算法、环境、配置或实验代码；唯一新增内容是本 Markdown 审计文件。
