# MAPPO短训练前检查——第六部分：Reward与约束

## 0. 检查范围与结论摘要

本次检查只覆盖 Reward、约束及其进入双 Agent MAPPO 的实际调用链。未修改 Reward、环境、Bridge、MAPPO、配置或物理模型；未联网、未安装依赖、未运行训练，也未触发任何 policy update。

检查方法包括：

- 静态追踪 `IDCPriceEnv20D.step()` 到 HARL runner 的真实数据流；
- 核对正式配置、Reward 计算式、归一化尺度与约束实现；
- 使用正式环境工厂和固定 seed `7110`，串行运行 5 组独立的 24 步固定动作轨迹；
- 每组均重新创建环境，且任务、服务器和到达过程完全一致；
- 固定动作验证共执行 120 个环境步，使用真实 AC OPF/MEF 路径，但不含 actor、critic、buffer 或参数更新。

核心结论：

1. 当前基础 Reward 的符号、求和、终止项和双 Agent 共享传递均正确，未发现 NaN/inf、重复累加或漏项。
2. 低负载策略不会通过“少做任务”获得更高回报；在固定场景中，高负载策略明显优于中、低负载策略。
3. Reward 的主要量级由队列、溢出、终止积压、完成工作和碳排放构成；多个 backlog 项存在有意但较强的重叠塑形。
4. BESS 充放电方向与成本、碳排放、SOC 的物理响应一致；但持续请求 SOC/功率边界外的“期望功率”会产生较大的 `invalid_action` 软惩罚，即使输入动作本身仍在合法 Box 内。
5. 当前 Grid/Safe 指标只进入 observation 和 `info`；正式主配置中 Grid penalty 关闭，Safe cost 不进入 PPO 目标，也没有 Safe-RL cost buffer/Lagrange 更新。
6. 主 MAPPO 短训练不存在 Reward 级硬阻塞，但开始后必须监测各分量占比；`no_bess` 消融配置存在终止 SOC 惩罚仍生效的问题，不应在修正前用于结论性消融。

---

## A. 当前 Reward 调用链

```text
IDCPriceEnv20D.step(action_23d)
  文件：envs/idc_price_env.py
  输出：base_reward: scalar，info: Reward分量和物理量
        │
        ▼
GridCoupledEnv.step(action_23d)
  文件：env_wrappers/grid_coupled_env.py
  计算：grid_adjusted_reward = base_reward - grid_penalty
  当前正式配置：grid_penalty == 0
        │
        ▼
IDCGridMultiAgentEnv.step({idc, bess})
  文件：marl/envs/idc_grid_multi_agent_env.py
  操作：将同一个团队标量 Reward 复制给 IDC、BESS 两个 Agent
        │
        ▼
HarlIDCGridBridge.step(actions)
  文件：marl/bridges/harl_bridge.py
  输出：rewards.shape == (2, 1)，dtype=float32
        │
        ▼
HarlPaddedBridge.step(padded_actions)
  文件：marl/bridges/harl_padded_bridge.py
  操作：动作裁剪维度/还原后调用底层 Bridge；Reward 原样传递
        │
        ▼
ShareDummyVecEnv.step_wait()
  HARL文件：harl/envs/env_wrappers.py
  输出：rewards.shape == (n_rollout_threads, 2, 1)
        │
        ▼
OnPolicyBaseRunner.insert(data)
  HARL文件：harl/runners/on_policy_base_runner.py
  EP centralized critic只插入 rewards[:, 0]
        │
        ▼
OnPolicyBaseRunner.compute()
  HARL文件：harl/runners/on_policy_base_runner.py
  对单一团队Reward计算return/GAE
        │
        ▼
OnPolicyMARunner.train()
  HARL文件：harl/runners/on_policy_ma_runner.py
  将同一份共享advantage分别交给IDC actor和BESS actor；critic只更新一次
```

不存在以下问题：

- 两个 Agent 的 Reward 在 centralized critic 中求和两次；
- Padded Bridge 再次缩放或累加 Reward；
- actor buffer 各自重复存一套 Reward 并导致团队 Reward 加倍；
- Grid wrapper 在 Grid penalty 为零时改变基础 Reward。

`BaseLogger` 对两个 Agent 的相同 Reward 取均值，因此 episode reward 也不会翻倍。

---

## B. 完整 Reward 公式与配置

### B.1 基础 Reward

记归一化量为带帽变量，当前每步基础 Reward 为：

\[
\begin{aligned}
r_t ={}& 5.0\hat W_t
+1.5\hat N^{finish}_t
+0.6\hat P^{finish}_t\\
&-0.35\hat C_t
-0.30\hat E_t
-0.80\hat Q_t
-1.20\hat Q^{overflow}_t\\
&-0.80\hat Q^{urgent}_t
-0.25\hat T^{wait}_t
-1.20\hat N^{deadline}_t
-0.80\hat S_t\\
&-0.08\hat U_t
-1.00\hat P^{peak}_t
-0.15\hat N^{pause}_t
-0.03\hat N^{resume}_t\\
&-0.80\hat N^{nonint}_t
-0.05\Delta L_t
-0.03\Delta A_t
-1.00\hat C^{degradation}_t\\
&-0.20\hat A^{invalid}_t
+\mathbb{1}_{terminal}\left(-3.0\hat Q_T-2.0\hat S_T^{excess}\right).
\end{aligned}
\]

源码位置：`envs/idc_price_env.py` 的 `IDCPriceEnv20D.step()`。

### B.2 分量、权重与实际含义

| `info`字段 | 权重 | 方向 | 实际含义 |
| --- | ---: | --- | --- |
| `r_done` | +5.00 | 奖励 | 本步完成的工作量；名称不是 episode done |
| `r_finished_tasks` | +1.50 | 奖励 | 本步完成的任务数量 |
| `r_priority` | +0.60 | 奖励 | 本步完成任务的优先级和 |
| `r_cost` | -0.35 | 惩罚 | 电费 |
| `r_carbon` | -0.30 | 惩罚 | 碳排放 |
| `r_queue` | -0.80 | 惩罚 | 当前积压工作量 |
| `r_overflow` | -1.20 | 惩罚 | 超出软队列容量的积压 |
| `r_urgent` | -0.80 | 惩罚 | 紧急或高优先级任务积压 |
| `r_waiting` | -0.25 | 惩罚 | 活跃未完成任务平均等待时间 |
| `r_deadline` | -1.20 | 惩罚 | 本步首次越过 deadline 的任务数 |
| `r_sla` | -0.80 | 惩罚 | 已逾期活跃任务的优先级加权延迟 |
| `r_unused` | -0.08 | 惩罚 | 计划容量未被任务实际使用的部分 |
| `r_peak_load` / `r_grid_peak` | -1.00 | 惩罚 | 超过 `grid_power_limit_kW` 的购电功率；二者为同一值 |
| `r_pause` | -0.15 | 惩罚 | 已开始但本步暂停的任务数 |
| `r_resume` | -0.03 | 惩罚 | 暂停后恢复的任务数 |
| `r_noninterruptible` | -0.80 | 惩罚 | 非可中断任务被暂停的次数 |
| `r_load_smooth` | -0.05 | 惩罚 | 服务器负载动作变化 |
| `r_action_smooth` | -0.03 | 惩罚 | 任务偏好/BESS等动作变化 |
| `r_degradation` | -1.00 | 惩罚 | BESS 吞吐量导致的退化成本 |
| `r_invalid_action` | -0.20 | 惩罚 | 请求充放电功率与物理可执行功率之差 |
| `r_final_queue` | -3.00 | 终止惩罚 | 终止时剩余积压 |
| `r_final_soc` | -2.00 | 终止惩罚 | 终止 SOC 超出目标容差的部分 |

`r_grid_peak` 是为了诊断保留的 `r_peak_load` 别名。总 Reward 只加入一次 `r_peak_load`，不能在外部聚合时将二者再次相加。

### B.3 Grid调整

`GridCoupledEnv` 的最终标量为：

\[
r_t^{grid}=r_t^{base}-p_t^{grid}.
\]

当前正式配置中：

- `grid_reward.enabled = false`；
- `grid_reward.mode = none`；
- LMP、MEF、安全相关权重均为 `0`；
- 因而固定动作验证的 120 步中 `grid_penalty` 全部为 `0`。

`info["reward_total"]` 是基础环境 Reward；`info["grid_adjusted_reward"]` 才是 wrapper 最终返回值。目前二者相等。

### B.4 配置中存在但当前公式未使用的旧字段

| 配置项 | 当前状态 | 当前真正使用的替代项 |
| --- | --- | --- |
| `reward_peak_load_weight` | 被保存但未进入现行 Reward 公式 | `reward_grid_peak_weight` |
| `peak_power_threshold_kW` | 未用于当前峰值惩罚阈值 | `grid_power_limit_kW` |

这两个字段容易造成配置误读，应在后续配置整理中标记为 legacy，但本次未修改。

---

## C. 符号、方向与归一化

### C.1 实际归一化尺度

正式配置经过 IDC 缩放后的关键量为：

| 尺度 | 最终值 | 用途 |
| --- | ---: | --- |
| `queue_ref` | 600,000 | 完成量、积压、紧急积压、未使用容量 |
| `queue_capacity_ref` | 600,000 | 溢出起点 |
| 初始积压 | 30,000 | episode 初始任务 |
| `cost_ref` | 6,000 | 电费和退化成本 |
| `carbon_ref` | 1,500 | 碳排放 |
| `sla_ref` | 5,000 | SLA延迟 |
| `peak_power_ref` | 1,000 kW | 峰值超额 |
| `grid_power_limit_kW` | 1,800 kW | 当前峰值阈值 |
| BESS容量 | 10,000 kWh | SOC与可用能量 |
| 最大充/放电功率 | 2,000 kW | BESS动作和 invalid 归一化 |

### C.2 方向审计

| 行为变化 | 预期方向 | 固定动作结果 | 判断 |
| --- | --- | --- | --- |
| 完成更多任务 | Reward上升 | 高负载完成量最高且总回报最好 | 正确 |
| 积压/溢出增加 | Reward下降 | 低负载积压最大且总回报最差 | 正确 |
| 电费增加 | Reward下降 | 分量始终为负 | 正确 |
| 碳排增加 | Reward下降 | 分量始终为负 | 正确 |
| 超过购电上限 | Reward下降 | 充电策略出现峰值惩罚 | 正确 |
| SOC偏离目标容差 | 终止Reward下降 | 强充/强放均产生 `-0.7` | 正确 |
| 请求不可执行的BESS功率 | Reward下降 | 强充/强放累计显著 invalid 惩罚 | 方向正确，语义需关注 |

所有 Reward 与分量在 120 个固定动作环境步中均为有限值；分量逐步求和与 `reward_total` 的最大误差为 `0.0`。

---

## D. 五组固定策略的24步结果

### D.1 策略定义

| 策略 | IDC服务器动作 | BESS有效动作 | 含义 |
| --- | ---: | ---: | --- |
| A 中负载+空闲BESS | 20维均为0.50 | 0.50 | 中等服务器计划负载，BESS空闲 |
| B 低负载+空闲BESS | 20维均为0.01 | 0.50 | 极低服务器计划负载，BESS空闲 |
| C 高负载+空闲BESS | 20维均为1.00 | 0.50 | 最大服务器计划负载，BESS空闲 |
| D 中负载+持续充电 | 20维均为0.50 | 0.00 | BESS持续请求最大充电 |
| E 中负载+持续放电 | 20维均为0.50 | 1.00 | BESS持续请求最大放电 |

其余 IDC 偏好动作固定为 `0.50`。HARL侧使用 padded `(1, 2, 22)` 动作；五组均由正式环境工厂创建，使用相同 seed 且分别 reset。

### D.2 Episode汇总

| 指标 | A 中负载 | B 低负载 | C 高负载 | D 充电 | E 放电 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Episode Reward | -26.0724 | -48.7759 | **-12.4807** | -34.2993 | -30.5962 |
| 平均每步Reward | -1.0864 | -2.0323 | **-0.5200** | -1.4291 | -1.2748 |
| 最小Reward | -6.1177 | -9.8242 | -3.3180 | -7.0177 | -7.0177 |
| 最大Reward | -0.0394 | -0.1521 | **0.1194** | -0.2634 | 0.0249 |
| 完成工作量 | 374,076.97 | 7,481.54 | **686,798.35** | 374,076.97 | 374,076.97 |
| 工作量完成率 | 30.94% | 0.62% | **56.81%** | 30.94% | 30.94% |
| 完成任务数/31 | 16 | 0 | **23** | 16 | 16 |
| 终止积压 | 834,875.80 | 1,201,471.23 | **522,154.42** | 834,875.80 | 834,875.80 |
| Deadline miss | 5 | 19 | **1** | 5 | 5 |
| SLA violation任务 | 4 | 19 | **0** | 4 | 4 |
| 电费 | 14,206.30 | **7,463.31** | 19,362.21 | 15,679.99 | 12,876.30 |
| 碳排放 | 13,834.99 | **7,819.59** | 18,138.08 | 16,782.36 | 11,174.99 |
| 峰值超额累计 | 1,060.33 | 654.20 | 1,386.99 | **3,029.88** | 1,060.33 |
| 终止SOC | 0.50 | 0.50 | 0.50 | 0.90 | 0.10 |
| BESS吞吐量 | 0 | 0 | 0 | 4,210.53 | 3,800.00 |

低负载虽然降低了电费和碳排，但积压、溢出、紧急任务、deadline和终止积压的惩罚远大于节省量。因此未发现“通过不执行任务刷高 Reward”的明显漏洞。

### D.3 各Reward分量的episode累计值

| 分量 | A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| `r_done` | +3.1173 | +0.0623 | +5.7233 | +3.1173 | +3.1173 |
| `r_finished_tasks` | +0.7742 | 0 | +1.1129 | +0.7742 | +0.7742 |
| `r_priority` | +0.1512 | 0 | +0.1920 | +0.1512 | +0.1512 |
| `r_cost` | -0.8287 | -0.4354 | -1.1295 | -0.9147 | -0.7511 |
| `r_carbon` | -2.7670 | -1.5639 | -3.6276 | -3.3565 | -2.2350 |
| `r_queue` | -15.0700 | -21.1799 | -10.4362 | -15.0700 | -15.0700 |
| `r_overflow` | -5.0542 | -11.4391 | -0.5888 | -5.0542 | -5.0542 |
| `r_urgent` | -1.1462 | -5.6292 | -0.3407 | -1.1462 | -1.1462 |
| `r_waiting` | -0.8596 | -1.3661 | -0.6493 | -0.8596 | -0.8596 |
| `r_deadline` | -0.1935 | -0.7355 | -0.0387 | -0.1935 | -0.1935 |
| `r_sla` | -0.0064 | -0.4688 | -0.0002 | -0.0064 | -0.0064 |
| `r_unused` | 0 | 0 | -0.0082 | 0 | 0 |
| `r_peak_load` | 0 | 0 | 0 | -2.4578 | 0 |
| `r_pause` | 0 | 0 | 0 | 0 | 0 |
| `r_resume` | 0 | 0 | 0 | 0 | 0 |
| `r_noninterruptible` | 0 | 0 | 0 | 0 | 0 |
| `r_load_smooth` | -0.0150 | -0.0003 | -0.0660 | -0.0150 | -0.0150 |
| `r_action_smooth` | 0 | -0.0128 | -0.0130 | -0.0007 | -0.0007 |
| `r_degradation` | 0 | 0 | 0 | -0.0140 | -0.0127 |
| `r_invalid_action` | 0 | 0 | 0 | -4.3789 | -4.4200 |
| `r_final_queue` | -4.1744 | -6.0074 | -2.6108 | -4.1744 | -4.1744 |
| `r_final_soc` | 0 | 0 | 0 | -0.7000 | -0.7000 |
| **合计** | **-26.0724** | **-48.7759** | **-12.4807** | **-34.2993** | **-30.5962** |

### D.4 分量量级占Episode Reward绝对值的比例

比例定义为 `abs(分量累计值) / abs(Episode Reward)`。正负项相互抵消，所以各行之和允许超过 100%。

| 主要分量 | A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完成工作 | 12.0% | 0.1% | 45.9% | 9.1% | 10.2% |
| 完成任务 | 3.0% | 0% | 8.9% | 2.3% | 2.5% |
| 电费 | 3.2% | 0.9% | 9.1% | 2.7% | 2.5% |
| 碳排放 | 10.6% | 3.2% | 29.1% | 9.8% | 7.3% |
| 队列 | **57.8%** | **43.4%** | **83.6%** | **43.9%** | **49.3%** |
| 溢出 | 19.4% | 23.5% | 4.7% | 14.7% | 16.5% |
| 紧急积压 | 4.4% | 11.5% | 2.7% | 3.3% | 3.7% |
| 平均等待 | 3.3% | 2.8% | 5.2% | 2.5% | 2.8% |
| Deadline | 0.7% | 1.5% | 0.3% | 0.6% | 0.6% |
| 峰值购电 | 0% | 0% | 0% | 7.2% | 0% |
| Invalid BESS | 0% | 0% | 0% | 12.8% | 14.4% |
| 终止积压 | 16.0% | 12.3% | 20.9% | 12.2% | 13.6% |
| 终止SOC | 0% | 0% | 0% | 2.0% | 2.3% |

队列相关项是当前最主要的梯度信号。它们能防止“低能耗但不做任务”，但也意味着权重调整时不能只单独观察 `r_queue`，必须联合观察 `r_overflow`、`r_urgent`、`r_waiting` 和 `r_final_queue`。

---

## E. 单步Reward分解验证

以下取策略 A 的第 0、12、23 小时。数值均直接来自底层 `info`。

### E.1 第0小时

物理量：完成工作 `15,586.54`，积压 `14,413.46`，电费 `360.46`，碳排 `720.91`，购电功率 `1,029.88 kW`，SOC `0.50`。

| 分量 | 归一化量 | Reward贡献 |
| --- | ---: | ---: |
| 完成工作 | 0.025978 | +0.129888 |
| 电费 | 0.060076 | -0.021027 |
| 碳排放 | 0.480610 | -0.144183 |
| 队列 | 0.024022 | -0.019218 |
| 初始负载平滑 | 0.300000 | -0.015000 |
| **总计** |  | **-0.069540** |

### E.2 第12小时

物理量：完成工作 `15,586.54`，积压 `396,729.91`，电费 `598.29`，碳排 `256.41`，购电功率 `569.80 kW`。

| 分量 | Reward贡献 |
| --- | ---: |
| 完成工作 | +0.129888 |
| 完成任务 | +0.048387 |
| 完成优先级 | +0.008297 |
| 电费 | -0.034900 |
| 碳排放 | -0.051282 |
| 队列 | -0.528973 |
| 等待时间 | -0.024740 |
| **总计** | **-0.453323** |

### E.3 第23小时（终止步）

物理量：完成工作 `15,586.54`，终止积压 `834,875.80`，新 deadline miss `1`，SLA累计量 `17.035`，SOC `0.50`。

| 分量 | Reward贡献 |
| --- | ---: |
| 完成工作 | +0.129888 |
| 电费 | -0.039171 |
| 碳排放 | -0.123969 |
| 队列 | -1.113168 |
| 溢出 | -0.469752 |
| 紧急积压 | -0.179492 |
| 等待时间 | -0.106250 |
| Deadline | -0.038710 |
| SLA | -0.002726 |
| 终止积压 | -4.174379 |
| **总计** | **-6.117728** |

这三步及全部 120 步的逐项求和均与底层总 Reward 精确一致。

---

## F. Terminal Reward审计

终止项只在第 24 个环境步，即 `t == 23` 时加入一次：

\[
r_{finalQ}=-3\cdot Q_T/queue\_ref
\]

\[
r_{finalSOC}=-2\cdot\max(|SOC_T-SOC_{target}|-SOC_{tol},0).
\]

核查结果：

- `r_final_queue` 未在普通步骤出现；
- `r_final_soc` 未在普通步骤出现；
- 终止步仍同时包含普通的队列、溢出、紧急和等待塑形，因此最终积压会被普通项和终止项共同惩罚；这是当前显式设计，不是程序重复加同一个字段；
- 策略 A/B/C/D/E 的终止步绝对值约占各自 episode Reward 绝对值的 23.5%、20.1%、26.6%、20.5%、22.9%，没有单步大到完全淹没整段轨迹；
- VecEnv 自动 reset 后，终止 `info` 仍保留第23小时数据和 `original_obs`，没有被 reset 后的第0小时信息覆盖；
- 终止后返回的零 observation 在 Reward 计算完成之后生成，不影响终止 Reward。

终止项量级较大但当前未达到明显失控程度。正式训练应记录终止项占比，避免未来任务负载或缩放变化使其主导全部梯度。

---

## G. 任务、SLA与中断约束

### G.1 任务执行

`IDCPriceEnv20D.step()` 只执行已到达且未完成任务；任务工作量不会超过任务剩余量。服务器动作决定计划容量，实际完成量受活跃任务和剩余容量共同限制。

### G.2 优先级与连续性

- IDC动作第20维控制紧急/优先级偏好；
- 第21维控制连续执行偏好；
- 环境据此对候选任务排序；
- 完成高优先级任务通过 `r_priority` 获得正奖励。

### G.3 Deadline与SLA

- deadline miss 在首次越过 `latest_finish` 时记录一次，并进入 `r_deadline`；
- 任务逾期后并不会被硬删除，仍允许继续完成；
- `r_sla` 对仍活跃的逾期任务按优先级和延迟持续计罚；
- completion rate 进入 `info`，但不直接作为单独 Reward 分量；完成工作量和完成任务数已分别提供信号。

### G.4 可中断与并行属性

- 非可中断任务并非硬约束：仍可被暂停，但会同时产生暂停和 `r_noninterruptible` 软惩罚；
- `parallelizable` 当前只进入 observation 特征，没有在执行器中形成硬并行约束；
- 固定中/高/低负载轨迹没有触发 pause/resume/noninterruptible 分量，因此这些分量仍需在更有针对性的任务序列中单独验证，但不构成本次主短训练的启动阻塞。

---

## H. 成本、碳、峰值与BESS

### H.1 电网功率顺序

环境中的实际顺序为：IDC任务负载 → BESS充放电 → PV抵消 → 购电功率。购电功率使用 `max(P_bus_net, 0)`，不允许通过向电网反送电获得负电费或负碳排。

### H.2 BESS动作映射

底层 BESS 标量动作 `a` 的含义为：

- `a < 0.5`：充电；
- `a == 0.5`：空闲；
- `a > 0.5`：放电；
- 内部原始控制量为 `2a - 1`。

充放电功率受额定功率、SOC边界、效率以及放电不超过 IDC 本地需求等硬限制。由于分支互斥，不会同时充放电。

### H.3 固定动作物理响应

与策略 A 相比：

- D 持续充电使终止 SOC 从 `0.5` 升至 `0.9`，电费、碳排和峰值购电均增加；
- E 持续放电使终止 SOC 从 `0.5` 降至 `0.1`，电费和碳排均下降；
- A/D/E 的任务完成量、任务数、积压和 deadline 完全相同，说明三组差异确实来自 BESS 链，而不是任务场景漂移；
- D/E 均产生 `-0.7` 的终止 SOC 惩罚，符合目标 `0.5±0.05` 的公式。

### H.4 `invalid_action`语义

`r_invalid_action` 惩罚的不是 Box 越界。它衡量“动作要求的理想充/放电功率”与“受SOC、额定功率和IDC负载限制后实际执行功率”的差值。

因此：

- D 在 SOC 已到上界后继续发送合法动作 `0.0`，仍累计 `-4.3789`；
- E 在 SOC 已到下界或本地负载不足时继续发送合法动作 `1.0`，仍累计 `-4.4200`。

这不是计算错误，但名称容易被误解为动作空间违规。其量级在 D/E 中达到 episode Reward 绝对值的 12.8%/14.4%，正式训练应单独记录，确认它是在学习“物理可执行请求”，而不是意外压制所有边界动作。

### H.5 `no_bess`消融风险

当前 `no_bess` 实验配置只把容量和充放电功率设为零，并未同步关闭终止 SOC 目标/权重。按现有公式，零容量会使报告 SOC 落到 `0`，终止时产生：

\[
-2\cdot(0.5-0.05)=-0.9.
\]

所以该消融并非“只移除BESS能力”，还会引入一个不可满足的终止惩罚。主 MAPPO 配置不受影响，但 `no_bess` 消融在修正前不能用于公平比较。

---

## I. 约束分类

| 约束 | 类型 | 实现位置/方式 | 是否进入Reward |
| --- | --- | --- | --- |
| HARL有界Box动作 | 硬约束/分布支持 | HARL bounded actor及项目adapter | 否，越界先被拒绝 |
| IDC/BESS动作shape | 硬校验 | Bridge、adapter、底层环境 | 否 |
| 服务器计划容量 | 硬物理限制 | `IDCPriceEnv20D.step()` | 间接进入完成量/unused |
| 任务到达时间 | 硬约束 | 未到达任务不参与执行 | 间接 |
| 任务剩余工作量 | 硬约束 | 完成量不超过remaining work | 间接 |
| 非可中断属性 | 软约束 | 允许暂停并计数 | `r_pause`、`r_noninterruptible` |
| 并行属性 | 仅状态特征 | 当前未形成执行硬约束 | 否 |
| 队列容量 | 软约束 | 允许超出 | `r_overflow` |
| Deadline | 软约束 | 允许逾期继续执行 | `r_deadline`、`r_sla` |
| BESS SOC上下界 | 硬物理约束 | 功率和能量裁剪 | 终止SOC及invalid间接反映 |
| BESS额定功率 | 硬物理约束 | 充放电功率裁剪 | invalid间接反映 |
| 禁止反送电 | 硬物理约束 | `P_grid=max(...,0)` | 成本/碳间接反映 |
| 购电峰值上限 | 软约束 | 允许超过 | `r_peak_load` |
| 电压/线路安全 | 监测约束 | OPF结果写入observation/info | 当前不进入主Reward |
| OPF初始化可用性 | 启动硬检查 | 正式工厂启动校验 | 否 |
| 运行期OPF/MEF失败 | 监测/降级 | `info`失败标记和安全指标 | 当前权重为0 |
| 24步horizon | 硬终止 | 环境时间索引 | 触发终止项 |

---

## J. Grid、OPF、MEF与Safe约束

### J.1 当前实际状态

Grid observation 中包含 LMP、MEF、最小电压、线路负载率、网损、安全裕度和 OPF 成功标志。`GridCoupledEnv` 的 `info` 还提供：

- 电压越限；
- 线路越限；
- OPF失败；
- `safe_violation_voltage`；
- `safe_violation_line`；
- `safe_violation_opf`；
- `safe_violation_total` / `safe_cost`。

但当前 MAPPO 配置下：

- `grid_reward.enabled=false`；
- Grid penalty 各权重为0；
- `safe_cost` 只在 `info`，不进入 on-policy cost buffer；
- 无 Lagrangian multiplier、constrained critic 或 Safe-MAPPO 更新；
- MAPPO 当前优化的是共享任务/能耗 Reward，而不是显式安全约束目标。

因此，现阶段可以称为“带电网观测与安全诊断的普通 MAPPO”，不能称为 Safe MAPPO。

### J.2 OPF/MEF异常

正式环境工厂会在启动阶段验证 OPF 能力；运行期 OPF/MEF 状态会写入 `info`。然而由于 Grid/Safe 权重为0，运行期异常不会自动改变 PPO 优化目标。必须依靠后续日志和训练停止条件避免异常被仅仅记录后忽略。

---

## K. 双Agent共享Reward与信用分配

当前是标准 cooperative shared-reward 结构：

- IDC和BESS收到完全相同的团队 Reward；
- `share_param=false`，因此两个 actor 参数独立；
- centralized critic 只使用一份团队 Reward；
- EP state 下只计算一套 return/GAE；
- 同一份 advantage 分别用于 IDC actor 和 BESS actor 更新；
- BESS虚拟21维动作只影响其策略分布诊断，不会创建额外 Reward。

工程实现没有重复累加，但共享 Reward 本身会带来信用分配难度：IDC 任务动作对队列项影响很强，而 BESS 主要影响成本、碳、峰值、SOC和invalid项。两者量级差异可能导致 BESS 的学习信号较弱或被队列信号淹没。这是算法/实验解释问题，不是当前数据流错误。

---

## L. 数值稳定性与尺度风险

### L.1 已确认稳定的部分

- 120个固定环境步中 Reward、observation和主要物理量均无 NaN/inf；
- Reward分量求和误差为0；
- Grid adjusted identity误差为0；
- 两个 Agent 的共享 Reward 差为0；
- `r_grid_peak`与`r_peak_load`差为0；
- 当前 bounded Box动作、buffer和evaluate_actions的一致性已由前序测试覆盖，本次未重新评价或修改。

### L.2 需要训练期监控的部分

1. 队列相关项高度相关且合计量级大，容易主导 advantage。
2. 正负大项相互抵消，例如高负载策略中 `|r_queue|/|R_episode|` 达83.6%，完成工作达45.9%，碳项达29.1%。
3. 当前 `use_valuenorm=false`，没有 ValueNorm/PopArt 对 return 尺度做在线归一化。
4. smoke中曾观察到较大的 critic loss；结合本次 episode return 为十几到几十的量级，这一现象并不足以单独证明数值错误，也不应在短训练前据此立即修改 loss 或归一化。
5. 正式短训练应先观察 critic loss、return、value prediction、gradient norm 和各Reward分量，再决定是否启用 ValueNorm。

结论：当前没有仅凭固定动作结果就必须调整 Reward 系数或 critic 归一化的证据。

---

## M. `info`与日志可观测性

### M.1 环境已提供的数据

底层及Grid wrapper的 `info` 已包含：

- 全部 `r_*` Reward分量；
- `reward_total`、`base_reward`、`grid_penalty`、`grid_adjusted_reward`；
- 完成工作、完成任务、任务完成率、积压、overflow、deadline、SLA；
- IDC功率、BESS充放电功率、SOC、吞吐量、退化成本；
- 购电功率、电费、碳排、PV使用量；
- OPF、LMP、MEF、电压、线路和Safe violation字段。

这些字段足以重构当前 Reward 并诊断主要约束。

### M.2 HARL默认logger的缺口

HARL `BaseLogger/GYMLogger` 当前主要记录：

- episode reward；
- actor训练指标；
- critic训练指标；
- average step reward。

其 `per_step()` 没有把环境 `infos` 中的 Reward分量、物理量和Safe字段持久化。因此“环境能提供”不等于“短训练日志已记录”。此问题属于后续日志接入职责，本次按限制未修改，也未扩展到完整日志系统。

---

## N. 问题清单

### N.1 阻塞主MAPPO短训练

无 Reward/约束计算级硬阻塞。主配置可以进入短训练流程。

### N.2 不阻塞，但短训练前或启动时必须明确

1. Grid/Safe约束当前不进入优化目标；实验名称和结论必须明确这是普通共享Reward MAPPO，而不是Safe MAPPO。
2. 默认HARL logger不持久化环境 `info` 分量；若短训练目标要求分析 Reward、BESS和物理指标，需要训练入口补充提取。
3. `reward_peak_load_weight`、`peak_power_threshold_kW` 是当前公式未使用的旧字段，配置说明中不得把它们当作最终生效值。
4. `r_grid_peak`和`r_peak_load`是别名，日志聚合时不得双计。
5. `invalid_action`实际表示BESS物理不可执行功率差，不是Box越界；其量级必须单独监测。

### N.3 必须在相关消融前修正

1. `no_bess` 配置仍保留不可满足的终止SOC惩罚；在修正前不能用于公平消融。
2. 若论文或实验要求并行任务属性构成真实约束，当前 `parallelizable` 只作为观察特征，不足以支持该结论。
3. 若研究目标要求硬非中断约束，当前实现是软惩罚，而非硬执行约束。

### N.4 可以保持现状

1. 基础Reward各项符号和归一化方向；
2. 终止积压和终止SOC公式；
3. IDC/BESS共享团队Reward的数据流；
4. BESS充放电的物理方向、SOC和功率硬边界；
5. Grid penalty关闭时基础Reward原样传递；
6. 当前先保持 `use_valuenorm=false`，在短训练获得实际曲线后再决策。

### N.5 暂不处理

1. Safe-RL cost buffer、Lagrangian critic或约束优化；
2. HAPPO/HGTA/Safe-RL算法改造；
3. Reward权重系统性搜索；
4. 物理模型、任务模型或BESS映射重构。

---

## O. 第六部分通过判断

### 基础Reward判断

**选择 2：基本通过，可以开始MAPPO短训练，但必须补齐训练期分量监控。**

理由：公式、符号、终止项、物理方向、共享传递和数值有限性均已验证；固定策略没有发现明显奖励漏洞。当前风险主要是队列相关项占比较大、BESS信用分配和日志可观测性，而不是 Reward 代码错误。

### Grid/Safe子判断

**同时符合 4：Reward本身正确，但Grid/Safe约束尚未进入优化目标。**

这不阻塞“普通MAPPO短训练”，但阻塞任何“已实现Safe MAPPO”或“策略受到显式电网安全约束优化”的结论。

---

## P. 最小建议

### P.1 开始主MAPPO短训练前

不建议修改基础Reward公式或权重。训练入口至少应记录：

- 每个 `r_*` 分量的step/episode累计；
- base、grid adjusted和grid penalty；
- queue、deadline、SLA、cost、carbon、peak；
- SOC、charge/discharge、degradation、invalid action；
- OPF/MEF成功率及Safe violation；
- return、value、critic loss、actor loss和gradient norm。

### P.2 短训练后再决定

根据3个seed、40次update的真实曲线再判断：

- 是否需要降低重叠backlog塑形权重；
- 是否需要启用ValueNorm；
- BESS相关信号是否被IDC队列信号淹没；
- `invalid_action`是否需要改名、拆分或重新归一化；
- 是否需要将Grid/Safe从诊断量升级为优化约束。

### P.3 特定实验前

- 使用 `no_bess` 前，必须同步处理终止SOC目标/权重；
- 声称硬非中断或任务并行约束前，必须核对并补足执行语义；
- 开始Safe-RL前，需要独立cost通路、buffer、critic/Lagrange更新和评估阈值，不能仅依赖当前 `info["safe_cost"]`。

### P.4 建议保持不动

- `envs/idc_price_env.py` 的现行基础Reward计算；
- `env_wrappers/grid_coupled_env.py` 的 base/grid adjusted 分层；
- 双Agent shared reward 传递；
- HARL bounded Box动作实现；
- 当前BESS动作映射与物理裁剪；
- MAPPO loss、GAE及centralized critic流程。

---

## 附录：关键文件与函数

| 职责 | 文件 | 类/函数 |
| --- | --- | --- |
| 基础Reward、任务、BESS、终止项 | `envs/idc_price_env.py` | `IDCPriceEnv20D.step()` |
| Grid adjusted Reward、OPF/MEF/Safe info | `env_wrappers/grid_coupled_env.py` | `GridCoupledEnv.step()` |
| 双Agent共享Reward | `marl/envs/idc_grid_multi_agent_env.py` | `IDCGridMultiAgentEnv.step()` |
| HARL原始双Agent桥 | `marl/bridges/harl_bridge.py` | `HarlIDCGridBridge.step()` |
| HARL padding桥 | `marl/bridges/harl_padded_bridge.py` | `HarlPaddedBridge.step()` |
| 正式环境工厂 | `marl/envs/harl_env_factory.py` | `make_harl_single_env()`、`make_harl_train_env()`、`make_harl_eval_env()` |
| 正式主配置 | `configs/config_ultimate.py` | `ENV_CONFIG`及Reward/Grid配置 |
| 短训练HARL配置 | `configs/harl_mappo_short.yaml` | MAPPO参数 |
| VecEnv | `C:/Users/bulio/Desktop/IDC/HARL/harl/envs/env_wrappers.py` | `ShareDummyVecEnv` |
| Reward插入与return | `C:/Users/bulio/Desktop/IDC/HARL/harl/runners/on_policy_base_runner.py` | `insert()`、`compute()` |
| 共享advantage与actor/critic训练 | `C:/Users/bulio/Desktop/IDC/HARL/harl/runners/on_policy_ma_runner.py` | `train()` |
| 默认训练日志 | `C:/Users/bulio/Desktop/IDC/HARL/harl/common/base_logger.py` | `per_step()`、`episode_log()` |
