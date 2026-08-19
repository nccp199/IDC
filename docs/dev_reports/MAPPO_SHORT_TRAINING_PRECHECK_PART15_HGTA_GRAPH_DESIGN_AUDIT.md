# 算法框架搭建——第十五部分：HGTA 图结构审计与候选方案设计

## A. 阶段目标与执行边界

本阶段只完成现有系统审计、294 维 centralized state 拆解、实体/特征/关系梳理、三套 HGTA 图候选、接口与兼容性设计。没有实现 HGTA/HGT/GAT/GNN，没有改动 Runner、Buffer、Actor、Critic 训练逻辑、环境、状态、动作、Reward 或 Mask，也没有运行训练或性能比较。

审计基准为当前工作区代码与 `main` experiment case。索引均采用 **0-based、两端闭区间**。报告中的“当前可用”表示可由当前训练时刻的 294 维 state 或固定、可版本化的环境 schema 得到；不把日志、OPF 临时对象或 episode 结束后才能知道的结果冒充训练输入。

本阶段计数：

| 项目 | 数量 |
| --- | ---: |
| 训练 update | 0 |
| 模型参数修改 | 0 |
| 正式环境/算法代码修改 | 0 |
| HGTA 实现文件 | 0 |
| 环境探查 step | 1（单次中性有界动作，仅改变内存中的探查环境） |
| 新增正式文件 | 1（本审计报告） |

工作树在本阶段开始前已有第 1–14 部分的未提交修改；本阶段没有覆盖或回退这些改动。

## B. 审计调用链

### B.1 真实状态路径

```text
IDCPriceEnv20D._get_obs()                         280
  = current global 6
  + task-pool aggregate 10
  + six server-group feature blocks 6*20 = 120
  + full-horizon forecast 6*24 = 144
        ↓
GridCoupledEnv._augment_obs()                    288
  = base obs 280 + aggregate grid obs 8
        ↓
IDCGridMultiAgentEnv._build_outputs()            294
  = wrapped obs 288 + supplemental 6
        ↓
HarlIDCGridBridge._shared_observations()
  = same state copied for IDC and BESS: (2, 294)
        ↓
HarlPaddedBridge
  = pads local BESS obs 164→288 only; shared state stays (2,294)
        ↓
HARL EP runner
  = selects share_obs[:, 0] and stores [T+1, workers, 294]
        ↓
OnPolicyCriticBufferEP feed-forward generator
  = flattens [T, workers, 294] → [batch, 294]
        ↓
MLPCentralizedCritic → HARL VCritic → scalar V(s)
```

证据：

- `envs/idc_price_env.py::IDCPriceEnv20D._get_obs`（1581–1628）明确构造 `6 + 10 + 6*N + 6*horizon`；`_get_task_pool_features`（1399–1484）、`_get_server_features`（1486–1536）、`_get_forecast_features`（1538–1579）给出各段顺序。
- `env_wrappers/grid_coupled_env.py::GridCoupledEnv._augment_obs`（381–386）将 8 维 grid obs 追加在 280 维之后。
- `marl/observations/global_state_builder.py::GlobalStateBuilder.build`（16–25）唯一操作是 `np.concatenate((raw, supplemental))`。
- `marl/envs/idc_grid_multi_agent_env.py::IDCGridMultiAgentEnv._build_outputs`（200–213）调用 IDC/BESS observation builder 和 state builder；`_build_supplemental_values`（176–198）定义补充量。
- `marl/bridges/harl_bridge.py::HarlIDCGridBridge._shared_observations`（93–101）向两个 agent 发布相同 state；`marl/bridges/harl_padded_bridge.py::_pad_observations`（131–141）只给 BESS 本地观测末尾补 124 个 0。
- `C:/Users/bulio/Desktop/IDC/HARL/harl/runners/on_policy_base_runner.py::warmup/insert`（269–283、342–460）在 EP 模式只取 `share_obs[:,0]`。
- `C:/Users/bulio/Desktop/IDC/HARL/harl/common/buffers/on_policy_critic_buffer_ep.py`（27–35、73–84、202–250）保存 `[episode_length+1,n_rollout_threads,294]` 并展开前两维。
- `marl/critics/critic_factory.py::MLPCentralizedCritic`（12–48）强制末维为 294；`build_critic`（64–86）对 HGTA 明确抛出 `NotImplementedError`。

### B.2 关键判断

1. 294 维在 `GlobalStateBuilder.build` 唯一生成，是两个数组拼接：288 维 wrapped observation 与 6 维 supplemental。
2. 它不是 IDC 288 维与 BESS 164/288 维观测的拼接；BESS 局部观测只是从同一源选择特征。
3. 两个 agent 在 bridge 层得到的是同一 294 向量的两份副本；EP Buffer 只保留第 0 份，因此不存在 agent 重复信息进入 MLP。
4. 294 维没有 padding。124 个兼容 padding 只存在于 BESS 本地 288 维观测的 `[164:288]`。
5. 没有显式 `action_t` 或动作序列。`prev_loads`、SOC/energy、四个功率量是动作作用后的状态记忆；在 `s_t` 中代表此前已完成 transition 的结果，不是未来泄漏。

## C. 294 维集中状态完整拆解

### C.1 完整索引表

| 起始 | 结束 | 长度 | 字段/模块 | 数据来源 | 物理意义 | 代码范围/理论范围 | 更新频率 | 已归一化 | 建议进入图 |
| ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 6 | current global | `_get_obs` | `T_amb/40, price/price_ref, lambda/lambda_ref, Q/queue_ref, sin(t), cos(t)` | 前四项按 reference 缩放但未统一 clip；时间 `[-1,1]` | 每 step | 部分 | 是，作为全局/上下文 |
| 6 | 15 | 10 | task-pool aggregate | `_get_task_pool_features` | waiting/running/finished/unfinished 数量比例；urgent/overdue work；平均剩余 deadline、priority；parallelizable/interruptible work | clip `[0,1.5]` | 每 step | 是 | 是，一个 TaskPool 节点 |
| 16 | 35 | 20 | server previous load | `prev_loads` | 20 个服务器组上一时刻实际负载 | `[0,1]` | 每 step | 是 | 是，逐组 |
| 36 | 55 | 20 | server capacity | `model.C_server` | 20 组算力容量除以当前最大容量 | `(0,1]` | 环境实例内固定 | 是 | 是，逐组 |
| 56 | 75 | 20 | server efficiency | `server_compute_efficiency` 或 `C_server/P_max` | 20 组能效除以当前最大能效 | `(0,1]` | 环境实例内固定 | 是 | 是，逐组 |
| 76 | 95 | 20 | server unit cost | `price*(P_max-P_idle)/C_server` | 20 组单位算力电费，再除以组内最大值 | `[0,1]`；当前正电价下公共 price 因子抵消，实际上通常随 step 不变 | 名义每 step，通常恒定 | 是 | 可用，但应做冗余消融 |
| 96 | 115 | 20 | server available | `1-prev_loads` | 20 组剩余可用负载 | `[0,1]` | 每 step | 是 | 可用，但与 16–35 精确互补 |
| 116 | 135 | 20 | estimated server temp | `(T_amb+15*prev_load)/80` | 简化估算温度，不是独立热模型测量 | clip `[0,1]` | 每 step | 是 | 可用，需标注“估算” |
| 136 | 159 | 24 | price forecast | `price_t/price_ref` | 整个 24 h 的已知分时电价 | 全 forecast 统一 clip `[-1.5,1.5]` | episode 内固定 | 是 | 是，全局 |
| 160 | 183 | 24 | ambient-temperature forecast | `T_amb/40` | 整个 24 h 环境温度 | clip `[-1.5,1.5]` | episode 内固定 | 是 | 是，全局 |
| 184 | 207 | 24 | arrival forecast | `lambda_t/lambda_ref` | 根据本 episode 任务生成的 24 h 到达工作量曲线 | clip `[-1.5,1.5]` | reset 后 episode 内固定 | 是 | 是，全局；需披露前瞻假设 |
| 208 | 231 | 24 | PV forecast | `pv_t/pv_ref_kw` | 整个 24 h 可用 PV 出力曲线 | clip `[-1.5,1.5]` | episode 内固定 | 是 | 是，PV/全局 |
| 232 | 255 | 24 | horizon sin | `sin(2πh/24)` | 24 h 位置编码 | `[-1,1]` | 永久固定 | 是 | 是，全局 |
| 256 | 279 | 24 | horizon cos | `cos(2πh/24)` | 24 h 位置编码 | `[-1,1]` | 永久固定 | 是 | 是，全局 |
| 280 | 287 | 8 | aggregate grid | `_build_grid_obs_from_info` | IDC-bus LMP、MEF+/MEF-、全网最小电压、最大 line loading、network loss、security penalty、OPF success | 前 7 项除 reference 后 clip `[-10,10]`；最后一项 `{0,1}` | reset 为零 IDC 负载 OPF；其后每 step | 是 | 是，但只能作为聚合/IDC-bus 特征 |
| 288 | 293 | 6 | supplemental | `SUPPLEMENTAL_FIELDS` | SOC、energy kWh、P_IDC kW、P_grid kW、charge kW、discharge kW | SOC 有物理界；其余 5 项为原始物理量、未归一化 | 每 step；reset 后四个功率为 0 | 否（除 SOC） | 是，先按固定物理 reference 归一化 |

`SUPPLEMENTAL_FIELDS` 的精确顺序来自 `marl/specs/agent_specs.py`（26–33）：

```text
288 bess_soc
289 bess_energy_kWh
290 P_IDC_kW
291 P_grid_kW
292 bess_charge_power_kW
293 bess_discharge_power_kW
```

细粒度索引规则如下，避免从分组描述中二次猜测：

| 索引 | 精确字段 |
| ---: | --- |
| 0 | current `T_amb/40` |
| 1 | current `price/price_ref` |
| 2 | current `lambda/lambda_ref` |
| 3 | current `Q/queue_ref` |
| 4 | current `sin(2πt/horizon)` |
| 5 | current `cos(2πt/horizon)` |
| 6–9 | waiting、running、finished、unfinished task count / total task count |
| 10–15 | urgent work、overdue work、avg deadline left、avg priority、parallelizable work、interruptible work |
| `16+i` | ServerGroup `i` previous load，`i∈[0,19]` |
| `36+i` | ServerGroup `i` capacity |
| `56+i` | ServerGroup `i` efficiency |
| `76+i` | ServerGroup `i` normalized unit cost |
| `96+i` | ServerGroup `i` available load |
| `116+i` | ServerGroup `i` estimated temperature |
| `136+h` / `160+h` / `184+h` | hour `h` price / ambient temperature / arrival forecast，`h∈[0,23]` |
| `208+h` / `232+h` / `256+h` | hour `h` PV / sine / cosine forecast |
| 280–287 | LMP、MEF+、MEF-、normalized min voltage、max line loading、network loss、security penalty、OPF success |
| 288–293 | SOC、energy、P_IDC、P_grid、charge power、discharge power |

### C.2 分段一致性

```text
6 + 10 + 20 + 20 + 20 + 20 + 20 + 20 + 24*6 + 8 + 6
= 6 + 10 + 120 + 144 + 8 + 6
= 294
```

- 无重叠：每段起点等于上一段终点 + 1。
- 无遗漏：首维为 0，末维为 293。
- 无兼容占位维：所有 294 维均由实际数组字段生成。
- 服务器布局是 **feature-major**，不是每个服务器组连续 6 维；Graph Builder 必须把六个 20 维 block 转置/重排成 `[20,6]`。

### C.3 更新、常量、冗余与可用性

- 每 step 真正动态：0–5 中的当前时刻量、6–15、16–35、96–135、280–293。
- 环境实例内固定：36–75；76–95 在当前正价格曲线下归一化后公共 price 因子消失，通常固定。
- episode 内固定：136–279；其中 184–207 在每次 reset 重新生成，其他曲线由当前 scenario/config 决定。
- 精确冗余：`available = 1-load`；`energy = SOC*capacity`。估算温度也由当前温度与 load 决定。
- reset 特殊值：280–287 来自 nominal-zero-IDC-load OPF，不是实际 IDC 初始注入；290–293 被显式设为 0。
- 无 per-task、per-bus 动态、per-line 动态、generator 动态数组。OPF 内部虽产生这些值，但 294 只保留 8 个聚合结果。
- **294 不含碳因子/碳强度**：底层环境在 `envs/idc_price_env.py`（234–236、480、608–640）使用 `carbon_factor_t` 计算 reward/info，但 `_get_obs`（1581–1628）和 `_get_forecast_features`（1538–1579）均未拼入它。294 也不含全网总负荷或 grid load scale；`P_grid_kW` 只是 IDC 本地净购电功率。
- 无风电信息进入 294，尽管底层环境存在 `wt_t` 占位字段。

### C.4 泄漏审计

| 信息 | 判断 | 理由 |
| --- | --- | --- |
| 上一 transition 的 P_IDC/P_grid/BESS 功率 | 无未来泄漏 | `IDCGridMultiAgentEnv.step` 在 action 执行完成后生成下一 decision state；它们在 `s_(t+1)` 中描述已经发生的结果 |
| grid 指标 | 无未来泄漏，但有时序语义需写清 | step 后的 grid obs 对应刚完成 transition 的 OPF；reset 是 nominal solve |
| 24 h price/T/PV | 设计上的已知 forecast | 当前代码明确将其定义为已知/可预测外生条件 |
| 24 h lambda | 条件性可接受 | 它由本 episode 全部任务对象生成；不含未来完成结果，但等于提前知道到达工作量曲线，论文必须声明 forecast/oracle 假设 |
| future queue/server load/task completion | 不存在 | `_get_forecast_features` 没有这些量 |
| episode 评估指标 | 不存在 | 只在 info/log 中，不进入 `_get_obs` 或 state builder |

### C.5 MLP 实际输入范围

seed 7110 reset 实测 `share_obs` 为 `float32 (2,294)`，全量 finite，最小值 `-1.0`，最大值 `5000.0`。最大值来自未归一化 `bess_energy_kWh`；reset 时 BESS 容量经 scale 后为 10000 kWh，SOC 0.5 对应 5000 kWh。当前 MLP 因 `use_feature_normalization=true` 在 HARL MLP base 输入处应用 LayerNorm（`HARL/harl/models/base/mlp.py` 50–65），但送到 Critic 边界的原始数值仍混合了约 `[-10,10]` 与数千量级，未来 Graph Builder 不应依赖跨类型单个 LayerNorm 隐式修正，应显式保存 feature-wise normalization schema。

## D. 环境实体清单

| 实体 | 当前数量 | 代码类/数据结构 | 独立状态 | 适合做节点 | 主要问题 |
| --- | ---: | --- | --- | --- | --- |
| IDC 物理/逻辑站点 | 1 | `IDCPriceEnv20D`；`GridCoupledEnv.idc_bus_idx` | 有聚合 Q、负载、功率 | 是 | 单站点下节点数只有 1 |
| IDC Agent | 1 | bridge 的 `idc` agent | 有 288 本地 obs、22 有效动作 | 通常否 | 它是控制角色，不是额外物理实体；与 IDC 节点并列会重复 |
| BESS 设备 | 1 | `IDCPriceEnv20D` 中 SOC/energy/power 字段 | 有 | 是 | pandapower 网络中没有独立 storage 元件 |
| BESS Agent | 1 | bridge 的 `bess` agent | 164 本地 obs、1 有效动作 | 通常否 | 与物理 BESS 节点重复；协同是 CTDE 语义，不等于物理边 |
| PV | 1 个聚合时间序列 | `pv_t` 数组及 step info | 仅聚合可用/使用/弃光量 | 可作为概念节点 | `net.sgen=0`，没有独立 pandapower 接入元件 |
| IEEE Bus | 14 | pandapower `net.bus` / `GridCase.bus_ids` | 静态属性完整；动态结果只在 OPFResult | 是 | 294 无逐 bus 动态值 |
| 输电 line | 15 | pandapower `net.line` | 静态端点/阻抗存在 | 更适合做边 | `GridCase.branch_ids` 只列 line，不含 transformer |
| transformer | 5 | pandapower `net.trafo` | 静态端点/参数存在 | 更适合做不同关系的边 | 不能漏掉，否则 IEEE-14 图断裂 |
| 外部电网 | 1 | pandapower `net.ext_grid`，bus 0 | 有静态接入；动态发电在 OPF 临时结果 | 候选 C 可做节点 | 294 无其动态出力 |
| generator | 4 | pandapower `net.gen`，bus 1/2/5/7 | 静态属性；动态出力临时存在 | 候选 C 可做节点 | 294 无逐机动态出力 |
| 固有 grid load | 11 | pandapower `net.load` | 静态 base load，经时序 scale | 一般作为 bus 静态特征 | 单独节点会显著增加 schema 而当前 state 不支持动态分解 |
| 服务器物理台 | 等效 2000 | group scaling 语义 | 无逐台状态 | 否 | 不能把 2000 台伪造成可观测节点 |
| ServerGroup | 20 | `IDCModel20D` 的 `N=20` 数组 | 每组 6 个 state 特征 | 是 | 20 个 group，每组代表 100 台，不是 20 台服务器 |
| Task 对象 | 当前 31 | `idc_model.task.Task` | 内存中有逐任务字段 | 候选 C 可做 | 294 中仅聚合；数量由 config 可变；需要新 graph output/buffer 契约 |
| TaskPool | 1 | `_get_task_pool_features` 10 维聚合 | 有 | 是 | 不能从 10 维恢复 profile/priority 分组节点 |
| Task profile | 当前 5 个 key（含 initial backlog） | `Task.profile_key` | reset 实例中可统计 | 当前不适合独立节点 | 294 没有逐 profile 统计；实测 A/B/C/D 分布随 seed 变化 |
| Global Context | 1 个逻辑上下文 | current global + forecast + aggregate grid | 有向量 | 是或作为 pooling 后 fusion | 不是物理实体，关系应标为信息边 |

运行时 reset 实测：31 个 Task（1 initial backlog + 30 随机任务）；20 个 ServerGroup，每组 100 台，等效 2000 台；14 bus、15 line、5 transformer、11 load、4 gen、1 ext_grid、0 sgen、0 storage。配置证据见 `configs/config_ultimate.py`（24–32、59–62、68–81、113–116）；创建证据见 `envs/idc_price_env.py`（104–128、404–420）与 `grid_model/ieee14_loader.py`（26–49）。

当前是 **单 IDC**。IDC 配置为 IEEE bus number 9；pandapower 内部 0-based bus index 为 8。BESS 和 PV 没有各自的 pandapower 元件，但 `envs/idc_price_env.py`（592–598）先在本地计算 `P_IDC + BESS charge - BESS discharge - PV used`，再由 `GridCoupledEnv.step`（220–240）把该净负荷注入 IDC bus 8。因此可说它们在模型语义上与 IDC 共址于 bus 9，但不可声称仓库已有独立 storage/sgen 连接记录。

price、temperature、lambda、time encoding、LMP 与 MEF 是特征/信号，不应为了增加节点类型而实体化。carbon factor 当前只在底层 reward/info 链存在，**不属于 294 训练输入**；要使用它必须视为状态契约变更，本阶段及候选 A/B 均禁止加入。

`Task.assigned_servers` 字段存在（`idc_model/task.py` 7–32），当前执行逻辑没有写入真实 task→server 绑定。因此不能据此创建分配边，也不能用全连接 task-server 边冒充物理/调度关系。

## E. 节点特征来源

### E.1 IDC 节点

| 特征 | 来源 | 当前可用 | 动/静态 | 归一化 | 泄漏风险 |
| --- | --- | --- | --- | --- | --- |
| Q 压力 | state[3]，`_get_obs.Q_norm` | 是 | 动态 | 是 | 无 |
| 任务池 10 维 | state[6:16] | 是 | 动态 | 是 | 无；更适合 TaskPool 节点 |
| ServerGroup 聚合统计 | state[16:136] 后做 mean/max | 是 | 混合 | 是 | 无；会损失 group 异质性 |
| P_IDC/P_grid | state[290:292] | 是 | 动态 | 否 | 无；需固定 reference |
| IDC-bus LMP/MEF | state[280:283] | 是 | 动态 | 是 | 无；只代表 IDC bus |

### E.2 BESS 节点

| 特征 | 来源 | 当前可用 | 动/静态 | 归一化 | 泄漏风险 |
| --- | --- | --- | --- | --- | --- |
| SOC | state[288] | 是 | 动态 | 物理比例 | 无 |
| energy | state[289] | 是 | 动态 | 否 | 无；与 SOC×capacity 冗余 |
| charge/discharge power | state[292:294] | 是 | 动态 | 否 | 无；描述前一 transition 结果 |
| capacity/max powers | config/环境固定字段 | schema 可用 | 静态 | 可作 reference | 无 |

### E.3 PV 节点

| 特征 | 来源 | 当前可用 | 动/静态 | 归一化 | 泄漏风险 |
| --- | --- | --- | --- | --- | --- |
| 24 h available forecast | state[208:232] | 是 | episode 内固定 | 是 | forecast 假设 |
| 当前 PV | 可由 current hour 与 24 h 曲线选择 | 可推导 | 动态选择 | 是 | 需稳定定义 hour 解码；不建议作为唯一值 |
| used/curtail | step info only | 不在 294 | 动态 | 否 | 若扩展到下一 state 可用；当前不得加入 |

### E.4 Grid Bus 节点

| 特征 | 来源 | 当前可用 | 动/静态 | 归一化 | 泄漏风险 |
| --- | --- | --- | --- | --- | --- |
| bus id、vn_kv、上下限 | `GridCase.raw_network.bus` | schema 可用 | 静态 | 需固定缩放 | 无 |
| base p/q load 聚合 | `raw_network.load` 按 bus 聚合 | schema 可用 | 静态基值 | 需固定缩放 | 无 |
| has_gen/has_ext_grid/has_idc | pandapower tables + config | schema 可用 | 静态 | binary | 无 |
| per-bus voltage/LMP | `OPFResult` 临时字典 | 294 不可用 | 动态 | 未进入 state | 不能在 state-rebuild 方案中加入 |
| 全网 min voltage/max loading/loss | state[283:287] | 可用但仅聚合 | 动态 | 是 | 应放 Global/IDC-bus，不得广播后声称为逐 bus 量 |

### E.5 ServerGroup 节点

| 特征 | 来源 | 当前可用 | 动/静态 | 归一化 | 泄漏风险 |
| --- | --- | --- | --- | --- | --- |
| previous load | state[16+i] | 是 | 动态 | 是 | 无 |
| capacity | state[36+i] | 是 | 静态 | 是 | 无 |
| efficiency | state[56+i] | 是 | 静态 | 是 | 无 |
| normalized unit cost | state[76+i] | 是 | 通常静态 | 是 | 无；与价格的绝对水平脱钩 |
| available | state[96+i] | 是 | 动态 | 是 | 无；与 load 冗余 |
| estimated temperature | state[116+i] | 是 | 动态/推导 | 是 | 无；不是测量值 |

### E.6 Task / TaskPool 节点

| 特征 | 来源 | 当前可用 | 动/静态 | 归一化 | 泄漏风险 |
| --- | --- | --- | --- | --- | --- |
| TaskPool 10 维 | state[6:16] | 是 | 动态 | 是 | 无 |
| 单任务 remaining work/status/deadline/priority | `Task` 内存对象 | 不在 294 | 动态 | 否 | 必须显式 graph state；不能训练时从 294 恢复 |
| profile/interruptible/parallelizable | `Task` 对象 | 不在 294（仅聚合 work） | 混合 | 否 | 同上 |
| assigned server | `assigned_servers` | 字段存在但当前无真实赋值语义 | 动态设想 | 不适用 | 伪造边风险最高 |

### E.7 Global Context

当前可直接使用 `state[0:6] + state[136:280] + state[280:288]`，共 158 维：当前温度/价格/到达/队列/时间、144 维 horizon forecast、8 维 aggregate grid。若 Q 已在 TaskPool、grid 指标已在 IDC 节点，应通过 feature ownership 表避免无意重复。

## F. 关系和边类型

### F.1 当前可真实定义的关系

| 源节点类型 | 关系 | 目标节点类型 | 方向 | 静/动态 | 代码数据来源 | 边特征 |
| --- | --- | --- | --- | --- | --- | --- |
| Bus | line_connected | Bus | 物理双向，消息图存双向 | 静态 | `net.line.from_bus/to_bus` | 可选静态 r/x/length/rating；首版可无 |
| Bus | transformer_connected | Bus | 物理双向，消息图存双向 | 静态 | `net.trafo.hv_bus/lv_bus` | 可选 tap/ratio/rating；首版可无 |
| IDC | attached_to | Bus 8 | 双向反关系 | 静态 | `idc_ieee_bus_number=9`→`idc_bus_idx=8` | 无或 attachment type |
| BESS | colocated_with/attached_to | Bus 8 | 双向反关系 | 静态推导 | 本地净负荷公式 + IDC 注入 bus 8 | 无；必须标注“模型语义共址” |
| PV | injected_into | Bus 8 | 双向反关系 | 静态推导 | 本地净负荷公式 + IDC 注入 bus 8 | 无 |
| ServerGroup | belongs_to | IDC | 双向反关系 | 静态 | `model.N=20` 属于单 IDC model | 无 |
| TaskPool | queued_at | IDC | 双向反关系 | 静态 | 单 IDC 的 `self.tasks` 与聚合函数 | 无 |
| Global | context_for | IDC | 双向信息边 | 静态 | Graph schema 设计 | 无；不是物理边 |

IEEE-14 拓扑实测为 15 条 line + 5 个 transformer。把每条物理连接存为两个 directed message edges 时是 `2*(15+5)=40`。`GridCase.branch_ids` 只包含 line（`grid_model/ieee14_loader.py` 46–49），GraphSchema 必须另读 `net.trafo`，否则拓扑不完整。

### F.2 不应直接作为边

- IDC 与 BESS 共享 reward/目标：这是协同语义，可用同一 centralized critic 表达；不能因此宣称存在物理 `coordinated_with` 边。若论文确需协同 relation，应明确它是控制/信息边并做消融。
- price、temperature 等同时影响多个实体：它们更适合 Global Context 或 pooling 后 fusion，不是实体间物理连接。carbon factor 也属于这种信号，但当前 294 根本没有该字段，只能作为未来状态契约变更讨论。
- task 与所有 ServerGroup 的“可能运行”关系：当前调度不是显式 binding；全连接会产生虚构边。
- LMP/MEF 与 Bus 的相关性：当前只有 IDC bus 数值，不能广播到 14 bus 后称作 nodal features。

## G. 全局与时间特征

| 处理方式 | 信息重复 | 参数/计算 | 结构清晰度 | 扩展性 | 工程难度 | 论文表达 |
| --- | --- | --- | --- | --- | --- | --- |
| 拼到所有相关节点 | 高，144 维 forecast 会被重复 14–39 次 | 输入投影大、显存高 | 较差 | 节点数增大时恶化 | 低 | 容易被视为普通 feature concatenation |
| Global Context 节点 | 一份上下文；通过 typed edge 传播 | 增加少量 relation/edges | 强 | 好 | 中 | 最符合异构关系叙事，但需定义信息边 |
| typed pooling 后再拼全局向量 | 无节点级重复 | 最省 | 强，物理图与外生量分开 | 好 | 低—中 | 方法清晰，但 Global 不参与 message passing |

建议默认候选不是立即定案：把短的当前 global/grid 量放 Global Context 节点，把 144 维 horizon forecast 在 typed pooling 后 fusion；这样避免超长特征支配所有 node projections。研究者若强调“时空上下文传播”，再选择把 forecast 投影成 Global 节点 embedding。

## H. 候选 A：最小物理能源图

### H.1 Schema

| 节点类型 | 数量 | 原始特征维度 | 特征 |
| --- | ---: | ---: | --- |
| IDC | 1 | 8 | Q、四个 task pressure 摘要、mean server load、归一化 P_IDC/P_grid |
| BESS | 1 | 4 | SOC、energy/capacity、charge/max、discharge/max |
| PV | 1 | 24 | 24 h normalized PV curve |
| Bus | 14 | 7 | 静态 vn_kv、p/q base load、has_gen、has_ext_grid、has_idc、bus-id encoding |
| Global | 1 | 158 或“14+144 后融合” | current global、forecast、aggregate grid |

总节点 18。关系：line-connected、transformer-connected、IDC/BESS/PV↔Bus8、Global↔IDC。双向 message edge 共 `40+2+2+2+2=48`，关系 triple 约 10 个（正反关系分开计）。首版无 edge attributes；以后可加入 line/transformer 静态参数，但需 schema version 升级。

### H.2 单步图示

```text
Global ⇄ IDC ⇄ Bus8 ⇄ IEEE-14 topology (15 lines + 5 transformers)
                 ⇅
               BESS
                 ⇅
                 PV
```

### H.3 数据流与 pooling

```text
state[294] + immutable IEEE14 schema
→ per-type feature extraction/projection
→ 1–2 层 relation-aware attention
→ type-wise pooling or key-node readout(IDC,BESS)
→ forecast/global fusion
→ MLP value head
→ V(s) [B,1]
```

优先比较 `IDC+BESS key-node readout`、type-wise pooling 后拼接、Global/CLS readout；不建议纯 sum pooling，因为 14 个 Bus 会在数值上主导单节点类型。

### H.4 公平性、优缺点与风险

- Actor、HAPPO 顺序更新、采样、Reward、GAE 全部不变，只替换 centralized value function。
- 参数量通过共享 type projection、关系参数共享、较小 hidden range 控制；正式实验应同时报告原 MLP 和 parameter-matched wider MLP。
- 优点：固定规模；可完全由 294 + 静态 topology 重建；Buffer 无需改；最易做首个 smoke/gate。
- 缺点：不表达 20 个 ServerGroup，丢掉当前 state 中最清晰的结构性异质信息；14 个 Bus 只有静态节点特征，动态 grid 仍是聚合量。
- 工程风险：把 aggregate grid 指标误作 14 个 nodal 值；漏 transformer；设备共址语义写成已有 pandapower 元件。
- 科研风险：容易被审稿人认为只是对小物理图使用普通 GAT，HGTA 相对 MLP 的区别不充分。
- 预计工作量：低—中。

## I. 候选 B：计算—能源双层异构图

### I.1 Schema

在候选 A 上增加当前 state 真正支持的 20 个 ServerGroup 和 **一个** TaskPool；不创建无法从 294 恢复的 TaskGroup/profile 节点。

| 节点类型 | 数量 | 原始特征维度 | 特征 |
| --- | ---: | ---: | --- |
| IDC | 1 | 2–4 | 归一化 P_IDC/P_grid，可选 Q/IDC-bus LMP |
| ServerGroup | 20 | 6 | load、capacity、efficiency、unit cost、available、estimated temp |
| TaskPool | 1 | 10 | state[6:16] 原样 |
| BESS | 1 | 4 | 同候选 A |
| PV | 1 | 24 | 同候选 A |
| Bus | 14 | 7 | 同候选 A 静态 schema |
| Global | 1 | 14，另将 144 维 forecast 后融合 | current global 6 + aggregate grid 8 |

总节点 `1+20+1+1+1+14+1=39`。关系及 directed edges：

| 关系 | Directed edges |
| --- | ---: |
| Bus↔Bus lines/transformers | 40 |
| IDC↔Bus8 | 2 |
| BESS↔Bus8 | 2 |
| PV↔Bus8 | 2 |
| 20 ServerGroup↔IDC | 40 |
| TaskPool↔IDC | 2 |
| Global↔IDC | 2 |
| 合计 | 90 |

正反关系分开时约 14 个 relation triples。首版 edge attributes 可全部为空；line/trafo 静态参数作为独立 ablation，避免一开始扩大 schema。

### I.2 单步图示

```text
  SG00  SG01 ... SG19
    \    |       /
      ⇄  IDC  ⇄ TaskPool
          ⇅
        Global
          ⇅
         Bus8 ⇄ IEEE-14 bus/line/transformer graph
        ↗    ↖
      BESS    PV
```

### I.3 数据流与 pooling

```text
share_obs [B,294]
  ├─ six feature-major blocks → ServerGroup [B,20,6]
  ├─ task aggregate → TaskPool [B,1,10]
  ├─ supplemental → IDC/BESS
  ├─ PV curve → PV
  ├─ static versioned schema → Bus nodes + edge_index
  └─ current/global/grid + horizon forecast → Global/fusion
→ type-specific Linear projections
→ 1–3 relation-aware attention blocks + residual + LayerNorm
→ type-level pooling (mean within ServerGroup/Bus, identity for singleton types)
→ concatenate typed summaries + forecast embedding
→ value MLP → scalar V(s)
```

推荐的默认研究范围是 2 层（1 层为低成本对照，3 层用于验证 oversmoothing/更远传播）、多头数与 hidden size 做小范围选择、residual + LayerNorm 保留、dropout 作为可选正则。relation-specific projection/attention 与 shared attention + relation embedding 都应作为实现选择，不在本阶段锁定具体超参数。

### I.4 公平性、优缺点与风险

- Actor、HAPPO update、Reward、采样和 Buffer 均保持不变；HGTA 只输出同 shape 的 `V(s)`。
- 这是三套方案中最能利用 **当前 294 的真实结构** 的方案：20 组计算资源形成多节点类型，TaskPool→IDC→Bus/BESS/PV 形成计算—储能—电网耦合路径。
- 固定 39 节点/90 directed edges，适合当前 2 workers；attention 成本随 sparse edges 近似线性增长，而不是节点全连接的平方增长。
- 缺点：TaskPool 仍是一个聚合节点；Bus 动态值缺失；Global/IDC/TaskPool 的 feature ownership 需要去重。
- 工程风险：feature-major server block 重排错误、schema order/hash 不稳定、raw supplemental 未归一化、hetero batching 的 graph offsets 错误。
- 科研风险：单 IDC 下跨 topology 泛化仍未验证；20 个 ServerGroup 的 capacity/efficiency 很多是静态特征，模型可能主要利用 type bias；attention 可解释性不能只靠可视化宣称。
- 预计工作量：中。

## J. 候选 C：完整多尺度动态图

### J.1 Schema

未来显式 graph state 方案：逐任务 Task（当前 reset 为 31、配置可变）、20 ServerGroup、14 Bus、4 generator + 1 ext-grid、IDC、BESS、PV、Global，当前规模约 74 节点；可进一步加入 11 个 grid load，但首版不建议。

潜在关系包括 Bus line/transformer、generator/ext-grid↔Bus、IDC/BESS/PV↔Bus8、ServerGroup↔IDC、Task↔IDC、Global context。仅使用当前可真实定义的这些关系，约 160 个 directed edges；**不加入 task↔server assignment**，除非以后调度器真实产生绑定。若对 Task 与 ServerGroup 做全连接“候选可运行”边，将额外产生 `31*20*2=1240` 条边并使图规模爆炸，而且当前没有真实性依据。

动态特征计划可读取 Task 对象与 `OPFResult` 的 per-bus voltage/LMP、per-line loading、per-generator dispatch，但它们当前不在 294，也未经过 Bridge/Buffer 序列化。因此候选 C 不能采用 state-only Graph Builder；必须由环境显式输出带 mask/pointer 的 graph state。

### J.2 单步图示

```text
Task00 ... TaskK ──queued_at──> IDC <──belongs_to── SG00 ... SG19
                                  |
                                  v
                         BESS/PV → Bus8 → IEEE topology ← generators/ext-grid
                                  |
                               Global
```

### J.3 数据流与 pooling

```text
environment objects + OPFResult
→ explicit HeteroGraphState at reset/step
→ Bridge validation and serialization
→ rollout buffer stores dynamic node features, masks, graph_ptr/edge_index
→ relation attention
→ hierarchical pooling: Task→IDC, ServerGroup→IDC, Bus→Grid
→ global fusion → V(s)
```

动态 Task 节点应使用 packed batching：按 type 拼接所有图的节点，`batch_index` 标明所属 sample，`graph_ptr` 记录边界，padding mask 仅作为备选。episode/time 两维仍应先按 HARL sampler 展开为 batch 样本，再对各样本的图做 pack；不能把 rollout 时间边误当成物理图边。

### J.4 公平性、优缺点与风险

- 优点：表达能力和未来多 IDC/跨 topology 潜力最高；逐任务与逐 bus 动态能形成真正的多尺度结构。
- 缺点：当前训练接口数据不足；需要环境、Bridge、Buffer、checkpoint schema 大改，不再是“只替换 Critic”的最小接入。
- 工程风险：动态实体排序、task ID 生命周期、done/reset 后 ragged graph、并行 worker 序列化、OPF 临时数据同步、resume 精确性。
- 科研风险：当前单 IDC、24 step、31 tasks 的问题规模可能不足以支撑复杂模型；过度建模/参数量/训练方差；新增信息会让与 MLP 294 baseline 的输入公平性失效。
- 预计工作量：很高。

## K. 候选方案对比矩阵

| 维度 | 候选 A | 候选 B | 候选 C |
| --- | --- | --- | --- |
| 当前数据支持度 | 高（294+静态 topology） | 高（294+静态 topology） | 低；需显式 graph state |
| 物理解释性 | 高 | 高 | 很高（实现后） |
| 异构性表达 | 中 | 高 | 很高 |
| 相对 MLP 的区别 | 中偏弱 | 强 | 很强，但输入不再公平 |
| 单 IDC 适配 | 高 | 高 | 中，可能过度设计 |
| 多 IDC 扩展 | 中 | 高 | 很高 |
| 跨 topology 潜力 | 中 | 高 | 很高 |
| 实现复杂度 | 低—中 | 中 | 很高 |
| 运行成本 | 低—中 | 中 | 高—很高 |
| Buffer 改动 | 0 | 0 | 必需且较大 |
| Checkpoint 复杂度 | 中 | 中 | 很高 |
| 论文创新表达 | 弱—中 | 强 | 强，但验证负担最大 |
| 过度设计风险 | 低 | 中 | 很高 |
| 主要数据损失 | server group 结构 | per-task/per-bus dynamic | 无（但需新增输入） |
| 首版可审计性 | 高 | 高 | 低 |

## L. HGTA Critic 与 Graph Builder 接口

### L.1 推荐的首版边界

不建议现在把 Runner 改成通用 `CriticBatch(graph=...)`。当前 HARL EP Buffer 已稳定提供 `[B,294]`；候选 A/B 可在 HGTA Critic 内部用确定性、无状态 Graph Builder 重建固定图：

```python
values, next_rnn_states = critic.get_values(
    share_obs,          # np.ndarray/Tensor [B, 294]
    rnn_states_critic,  # [B, recurrent_n, hidden]
    masks,              # [B, 1]
)
```

MLP 分支继续原样读取 `[B,294]`。HGTA 分支内部执行：

```text
HeteroGraphBatch = GraphBuilder.build(share_obs, immutable_schema)
HGTAEncoder(HeteroGraphBatch) → graph_embedding [B,H]
optional recurrent head(graph_embedding, rnn_states, masks)
value_head → values [B,1]
```

返回 shape 和 RNN state 契约与 `VCritic.get_values` 完全相同，因此 `compute_returns`、GAE、value loss 和 HAPPO actor update 不需要修改。

### L.2 GraphBuilder 规范

输入：

- `share_obs`: `float32 Tensor[B,294]`，device 与 Critic 一致；拒绝末维错误、NaN/Inf。
- `GraphSchema`: 不可变、版本化对象，含 IEEE topology、节点/关系顺序、静态 bus features、normalization references、schema hash。

输出建议：

```text
node_features[type] : float32 [sum_N_type, F_type]
edge_index[relation]: int64   [2, sum_E_relation]
edge_attr[relation] : float32 [sum_E_relation, F_edge] | None
batch_index[type]   : int64   [sum_N_type]
graph_ptr[type]     : int64   [B+1]
global_features     : float32 [B,F_global] | None
```

稳定顺序：`global, idc, task_pool, server_group, bess, pv, bus`；relation 使用完整 triple `(src_type, relation_name, dst_type)` 排序。节点 ID 是 type-local：singleton 为 0，ServerGroup 为 0–19，Bus 为 pandapower canonical bus index 0–13；不得依赖 Python dict 插入顺序。固定图 batch 通过对 edge index 加每个 sample 的 type-local node offset 复制，不能共享未偏移 index。

归一化：

- 已归一化 state 段保持原值，不二次 min-max。
- energy、P_IDC/P_grid、charge/discharge 使用配置锁定的 capacity/max-power/power reference；reference 与统计量写入 checkpoint。
- 静态 bus p/q、vn_kv 等使用训练 schema 的固定 reference；禁止按当前 batch 最大值归一化，否则 batch composition 会改变输入语义。
- dtype/device 在 GraphBuilder 最后一次性转换；索引必须 `torch.long`，特征/edge attrs 与 critic dtype 一致。

### L.3 从 state 重建与显式 graph state 的判断

| 项目 | 294 重建（候选 A/B） | 环境显式 graph（候选 C） |
| --- | --- | --- |
| Bridge/Buffer 改动 | 无 | 必须 |
| MLP/HGTA 切换 | 简单 | 双接口维护 |
| Resume | 容易；保存 schema/hash | 需保存 ragged graph/buffer 状态 |
| 可恢复信息 | ServerGroup、TaskPool、aggregate grid | 可含 Task/per-bus/per-line 动态 |
| 丢失信息 | per-task、per-bus dynamic、真实 assignment | 取决于 graph contract |
| 当前合理性 | 高 | 仅未来研究版本 |

结论：第一版 HGTA 应采用 294 重建，不把图逐 step 存入 Buffer；候选 C 只有在研究者明确接受扩大输入和改动范围时才使用显式 graph state。

### L.4 rollout、episode、recurrent 与 masks

- EP Buffer 原始 shape 是 `[T+1,W,294]`；feed-forward generator 已变成 `[T*W,294]`。GraphBuilder只处理这个扁平 batch，不需要知道 T/W。
- recurrent generator 同样最终产出展平的 state batch，但 RNN state 只在 chunk 起点传入。HGTA 只替代每步 feature encoder，masks 继续原样传给 recurrent head。
- 当前正式配置 `use_recurrent_policy=false`、`use_naive_recurrent_policy=false`（`configs/harl_mappo_short.yaml` 59–60）。首版可只验证 feed-forward，但接口应保留 RNN 参数并对未支持配置 fail-fast，不应静默忽略 masks。
- 输出必须是 `values [B,1]` 与更新后的 RNN state；returns、GAE 与 HAPPO 不依赖输入是 MLP 还是图。

## M. 参数量和复杂度分析

稀疏异构 attention 的主要成本近似由下列量共同决定：

```text
node type projections: Σ_type O(F_type * H)
relation projections:  O(R * H²)（若完全 relation-specific）
message/attention:      O(L * E * H * heads)
node transforms:        O(L * N * H²)
readout/value head:     O(type_count * H²)
```

| 指标 | A | B | C（当前规模估计） |
| --- | ---: | ---: | ---: |
| 节点 | 18 | 39 | 约 74 |
| directed edges | 48 | 90 | 约 160；若 task-server 全连则额外 1240 |
| relation triples | 约 10 | 约 14 | 约 16+ |
| type projections | 5 | 7 | 8+ |
| 相对当前 MLP 成本 | 小幅—中等增加 | 中等增加 | 高，且 ragged batching/序列化额外开销大 |
| 2-worker 额外开销来源 | 图构建与小 kernel 启动 | 20 SG + typed relations | 动态 pack、IPC/Buffer、更多小 kernel |

小图并不保证 GPU 更快：2 worker 下 sparse hetero operations 的 Python dispatch/kernel launch 可能比算术本身更显著。首版应缓存静态 edge index，批量向量化 state slicing，避免每 sample Python 循环。不得在本阶段承诺具体吞吐。

参数公平性：主对照保留现有 `[32,32]` MLP；同时准备 parameter-matched MLP，报告总 trainable parameters、每 update samples、optimizer/lr/epochs 完全一致。不要通过把 HGTA hidden size调得远大于 MLP 来获得不公平容量优势。

## N. Checkpoint、Resume 与评估兼容

未来 HGTA checkpoint 至少新增并严格校验：

```text
critic_type
graph_schema_version
node_type_order
relation_type_order
feature_schema_hash
normalization statistics/references
HGTA architecture config
graph builder version
static topology hash
```

候选 A/B 的图由 state + immutable schema 确定，不需要逐 step 保存图，也不需要在 rollout Buffer 中保存图。Checkpoint 保存 canonical schema metadata/hash；resume 时先重建 schema，再逐字段比对，任何 node/relation order、feature ownership、normalization 或 topology mismatch 都 fail-fast。

当前 checkpoint 已记录 `critic_type`、`method_id`、critic/model/optimizer state 并严格比较 compatibility（`marl/checkpointing/training_checkpoint.py` 197–216、254–310、390–447）。HGTA 应扩展同一机制，而不是绕过它。

MLP 与 HGTA checkpoint 不能互换：网络参数 key/shape、optimizer slots、feature schema、normalization 语义都不同。完整 training resume 必须拒绝跨 critic 加载。Actor 架构未变，因此可以提供明确的 **actor-only transfer** 模式，只载入 IDC/BESS actor state；这不等同于精确 resume，必须生成新 run manifest 并重置 critic/optimizer/update counter。

固定评估当前是 actor-only loader（`marl/evaluation/model_loader.py` 文件级契约）。因此：

- HGTA 不影响确定性 Actor 执行；标准 eval 不构图、不实例化 Critic。
- eval manifest 仍记录训练时 `critic_type=hgta`、`method_id=HAPPO_HGTA`、schema hash，便于溯源。
- value/attention 解释应作为独立 analysis mode，同时加载 critic 与 graph schema，不能污染标准固定评估路径或指标。

## O. 论文方法表达

候选 A：

> 在 CTDE 框架下，以 IDC、BESS、PV、IEEE-14 bus 和全局上下文构成类型化物理能源图，利用线路、变压器和设备接入关系估计集中式状态价值，同时保持两个分散 Actor 不变。

候选 B：

> 在 CTDE 框架下，以 ServerGroup–TaskPool–IDC 计算层和 IDC–BESS–PV–IEEE bus 能源层构成固定规模异构图，通过类型专属投影与关系注意力建模计算负载、储能和电网约束的耦合，仅替换 HAPPO 的集中式价值函数。

候选 C：

> 在 CTDE 框架下，以可变任务、服务器组、设备、发电单元和电网节点构成多尺度动态图，并以分层关系聚合估计集中式状态价值，分散 Actor 的执行接口保持不变。

论文风险判断：

1. 候选 B 最能支持“异构图”主张，因为它包含 7 类节点和真实的计算/组织/物理关系，而非仅在 14 bus 上做同构 attention。
2. 候选 A 最可能被认为只是带少量设备节点的普通 GAT；需要 relation/type ablation 才能证明 HGTA 必要性。
3. 候选 C 过度建模风险最高，且新增 per-task/per-bus 输入会破坏与 294 MLP 的输入公平性。
4. 单 IDC 下通过 20 个真实 ServerGroup、TaskPool 与能源层避免“类型太少”，但不能凭单场景证明多 IDC 泛化。
5. 若论文主张跨拓扑或规模泛化，需要后续 IEEE-30/57 或多 IDC 实验；当前候选设计只能说明接口有扩展能力，不能说明已实现或已验证。

## P. Codex 推荐方案

**推荐候选：候选 B——计算—能源双层固定异构图。**

推荐原因：它最大程度使用当前 294 中真实存在且可逐实体恢复的 20 个 ServerGroup，同时保留 TaskPool、IDC、BESS、PV 和 IEEE topology；固定 39 节点/90 directed edges 可在不改 Buffer 的前提下重建，研究表达也明显强于最小物理图。

依赖的事实：20 组各有 6 维特征；Task 只有 10 维 aggregate 可进当前 Critic；IEEE 拓扑静态可稳定提取；BESS/PV 与 IDC bus 的共址可由功率链确认；EP Buffer 始终保存 `[B,294]`。

主要风险：Bus 缺乏逐节点动态量；ServerGroup 有冗余/静态特征；Global、IDC 与 TaskPool 容易重复同一信号；单 IDC 不能独立证明跨拓扑价值。

不推荐候选 A 为论文主方案：它舍弃最有价值的计算层结构，异构性主张偏弱。可保留为低复杂度 ablation/工程 smoke。

不推荐候选 C 为第一版：它要求新增当前 294 不含的信息并改动 Bridge/Buffer/checkpoint，成本和输入公平性风险远高于当前阶段收益。可作为后续扩展路线。

研究者仍需决定最终节点/关系、Global 处理、pooling、是否保留冗余 ServerGroup 特征以及论文是否要求跨 topology；本报告的推荐不替代该决定。

## Q. 需要研究者确认的问题

1. 第一版是否选择候选 B，并明确限制为固定 20 ServerGroup + 1 TaskPool，而不创建逐任务节点？
2. 是否保留全部 14 个 Bus，即使当前只有静态 bus 特征和聚合 grid 动态量；还是用单一 Grid 聚合节点做更保守对照？
3. PV 是否作为独立概念节点，并在论文中明确 `net.sgen=0`、其接入关系来自本地净负荷计算而非 pandapower 元件？
4. 全局 price/time/forecast 使用 Global/CLS 节点，还是在 typed pooling 后 fusion；是否禁止把 144 维 forecast 复制到所有节点？当前 294 无 carbon factor，是否明确禁止首版擅自加入？
5. value readout 选择 type-wise pooling、Global/CLS，还是 IDC+BESS key-node readout；需要哪两种作为消融？
6. ServerGroup 的 `available=1-load`、`energy=SOC*capacity`、normalized unit cost 的冗余特征是否保留，并安排 feature ablation？
7. line 与 transformer 是否作为两个关系类型；首版是否加入阻抗/rating/tap edge attributes，还是先保持无 edge attrs？
8. 第一版是否只支持 feed-forward 配置并对 recurrent fail-fast，还是从开始就要求图 encoder + recurrent head 完整兼容？
9. 论文是否必须主张多 IDC/IEEE-30/57 泛化；若是，是否接受这需要后续环境和实验而非当前代码已有能力？
10. 公平性协议是否同时采用原 `[32,32]` MLP 与 parameter-matched MLP 两个 baseline，并固定所有 Actor/HAPPO/采样超参数？

## R. 问题分类

### R.1 阻塞 HGTA 设计

无阻塞候选 A/B 结构设计的问题。294 可精确映射，静态 IEEE topology 可提取，现有 Critic API 可保持。候选 C 被当前数据接口阻塞：per-task、per-bus/per-line dynamic 与 task-server assignment 不在 294。

### R.2 实现前必须决定

- 候选 A/B/C 的最终选择。
- Bus 是否全部建节点、Global/fusion 方式、readout、关系正反向定义、edge attributes。
- feature ownership 与冗余特征保留策略。
- recurrent 首版支持范围与 parameter-matched baseline 协议。

### R.3 实现阶段解决

- canonical schema/dataclass、vectorized block reshape、batched edge offsets、device/dtype、finite/shape assertions。
- schema/hash/checkpoint 字段、单元测试、固定 seed GraphBuilder determinism。
- HGTA 参数初始化、optimizer 接入、日志字段；这些不能改变 HAPPO 语义。

### R.4 后续实验解决

- HGTA 是否优于 MLP、参数公平性、吞吐/显存、attention 稳定性与解释性。
- relation/node/feature/readout ablation。
- 多 IDC、IEEE-30/57、跨拓扑/规模泛化。

## S. 第十五部分判断

| 项目 | 判断 | 说明 |
| --- | --- | --- |
| 294 维 state 可解释性 | 充分 | 全部 0–293 精确映射，无 unresolved 段 |
| 实体数据完整性 | 候选 B 充分 | ServerGroup/TaskPool/IDC/BESS/PV aggregate/Bus topology 可用；逐任务动态不在 state |
| 关系数据完整性 | 基本充分 | 物理/组织关系可定义；设备接入为功率链推导；无 task-server binding |
| 节点特征可用性 | 候选 B 充分但有冗余 | raw supplemental 需显式归一化；Bus dynamic 缺失 |
| Graph Builder 可实现性 | A/B 高，C 低 | A/B 从 294+immutable schema 重建；C 需新接口 |
| Buffer 兼容性 | A/B 完全兼容 | 保留 EP `[T+1,W,294]` |
| Checkpoint 兼容性 | 可设计 | 需新增 schema/order/hash/normalization metadata |
| 固定评估兼容性 | 高 | actor-only 标准 eval 不构图 |
| 候选方案可信度 | A/B 高，C 为未来设计 | 没有把缺失动态值伪装成现有数据 |

总体选择：**2. 基本充分，但有少量字段/方案需研究者进一步确认。**

这里的“少量”不是指 294 存在未解析索引，而是指 HGTA 研究选择：Global/forecast 的归属、Bus 静态节点是否值得保留、readout、冗余 feature 与 edge attributes。若研究者坚持候选 C，则总体判断改为“3. 当前 state 或接口不足，需先补数据”。

### S.1 运行时验证记录

成功 reset 探查使用的解释器、`PYTHONPATH` 与入口如下；实际运行的是同一 here-string 中更完整的 JSON 统计，下面是可直接复现关键 shape、相等性和拓扑计数的等价核心命令：

```powershell
$env:PYTHONPATH='C:\Users\bulio\Desktop\IDC\ultimate_simplify;C:\Users\bulio\Desktop\IDC\HARL'
@'
import numpy as np
from marl.envs.harl_env_factory import make_harl_single_env

env = make_harl_single_env(seed=7110)
try:
    obs, share_obs, available_actions = env.reset()
    true_obs = env.last_true_observations
    core = env.env.env
    grid = core.env
    base = grid.env
    net = grid.grid_case.raw_network
    print(obs.shape, share_obs.shape, [x.shape for x in true_obs])
    print(np.array_equal(share_obs[0], share_obs[1]))
    print(np.array_equal(obs[0], share_obs[0, :288]))
    print(np.all(obs[1, 164:] == 0))
    print(len(base.tasks), base.model.N, base.server_group_size)
    print(len(net.bus), len(net.line), len(net.trafo), len(net.sgen), len(net.storage))
finally:
    env.close()
'@ | & 'C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe' -
```

结果：exit code 0；`obs=(2,288)`、真实未 padding obs 为 `(288,)` 与 `(164,)`、`share_obs=(2,294)`、dtype float32、全 finite；两 agent state 相同；IDC obs 等于 state[0:288]；BESS `[164:288]` 全 0。reset state min/max 为 `-1/5000`。没有写临时文件。

单次 step 探查在同一命令链中执行 `actions=np.full((2,22),0.5,dtype=np.float32)` 与 `env.step(actions)`。BESS 有效动作 0.5 经 action adapter 映射为 raw 0（idle）；IDC 使用边界内中性探查值。该命令在 step 已完成后因审计脚本误读不存在的 `Task.profile` 字段而 exit code 1；随后只用 reset-only 脚本修正为真实字段 `Task.profile_key`，没有再次 step。两次失败都没有训练、backward、optimizer 或文件写入。最初一次错误 Python 路径 `idc_rl_env` 在解释器启动前失败，也未创建环境。

探查只改变了进程内环境对象，`finally: env.close()` 后释放；没有模型对象或优化器。总 step=1，总 update=0，临时文件=0。

### S.2 最终硬性确认

```text
训练 update 数量 = 0
模型参数修改 = 0
正式环境/算法代码修改 = 0
HGTA 实现文件 = 0
```

## T. 下一阶段唯一建议

请研究者先对 Q 节的 10 个问题作出选择，尤其确认“候选 B + 固定 20 ServerGroup + 1 TaskPool + 14 Bus + Global/forecast 处理 + readout”。确认前停止：不实现 HGTA，不修改生产代码，不运行训练或性能比较。
