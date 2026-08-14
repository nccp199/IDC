# Phase 2.7：25 MW IDC 正式化与训练尺度修复

日期：2026-08-14
结论：**READY FOR FORMAL TRAINING**

本报告只判断环境、输入语义、奖励尺度和三条正式算法入口是否具备开始正式训练的条件；它不是长训练结果，不构成 MAPPO/HAPPO/HGTA 性能结论，也没有引入 Safe RL。

## 1. 执行摘要

25 MW 已成为正式训练环境的单一 IDC 主尺度。正式配置为：

| 项目 | 正式值 |
|---|---:|
| `facility_rated_power_mw` | 25.0 MW |
| `server_group_size` | 1841 等效服务器/组 |
| `task_workload_scale` | 1841 |
| server groups | 20 |
| BESS | 2 MW / 10 MWh，未重新标定 |

正式额定定义是：在最大合法 compute action = 1、planned total load = 0.65、30 °C 高温参考条件下，`P_IDC = P_IT + P_cooling + P_others ≈ 25 MW`。确定性验收得到 **25.0013 MW**，绝对误差 0.0013 MW。

本阶段修复了四个训练尺度问题：SLA 惩罚不再随 workload scale 被稀释；BESS degradation 使用独立 BESS 物理参考；六个 supplemental 特征在 observation/state 构造层统一归一化一次；HGTA 只做结构映射，不再二次除以旧功率参考。

## 2. 正式 25 MW 配置与物理链

正式配置只在 `IDC_SCALE_CONFIG` 定义一次，环境构造继续沿用现有 config 传播链。没有增加 server-group 节点，没有改变 task forecast 机制，也没有把最终 `P_grid` 粗暴乘倍数。

物理链保持为：

`server action / planned load → per-group server load → IT power → cooling power → P_others → facility P_IDC`

额定点测试直接以 20 组异构 server model、每组 1841 台等效服务器、每台 `base_load + max_task_load_per_server` 和 30 °C 温度计算，结果为 25.0013359 MW。

固定 seed=2026 的 24 小时环境 smoke：

| case | server action | BESS | P_IDC min/mean/max (MW) | AC OPF |
|---|---:|---|---:|---:|
| normal | 0.5 | 3 h charge + 3 h discharge + 18 h idle | 18.3871 / 18.7834 / 19.2152 | 24/24 |
| high | 1.0 | idle | 20.0941 / 24.0816 / 25.0013 | 24/24 |

这与前一阶段 25 MW normal/high case 的物理量级一致；未测试其他 MW 尺度。

## 3. SLA reward scale 修复

旧公式把 SLA reference 乘以 workload scale：

`sla_norm_old = sla_penalty / (sla_ref × task_workload_scale)`

但 `sla_penalty` 表示 task count、priority 和 lateness hour，不是 MW 或 compute-work 数量。由 group size 100 变为 1841 时，同一 SLA 事件会被额外稀释 18.41 倍。

新公式使用独立固定参考：

`sla_norm_new = sla_penalty / sla_penalty_ref`，其中 `sla_penalty_ref = 50.0`。

合成相同数量、优先级和延迟的任务事件得到：

| group size | raw SLA penalty | reference | normalized penalty |
|---:|---:|---:|---:|
| 100 | 1000.859847 | 50.0 | 20.017197 |
| 1841 | 1000.859847 | 50.0 | 20.017197 |

reward 权重与 SLA 物理/事件定义均未改变，只修复 normalization。

## 4. BESS degradation reward scale 修复

旧公式为：

`r_deg_old = -w_deg × degradation_cost / IDC_scaled_cost_ref`

其中 `cost_ref` 随 IDC scale 放大，导致同一 2 MW / 10 MWh BESS 轨迹的 degradation 相对权重被稀释。

新参考只由 BESS 自身定义：

`bess_degradation_cost_ref = max_bess_power × timestep × degradation_cost_per_kWh`

当前值为 `2000 kW × 1 h × 0.02 = 40.0`，新公式为：

`r_deg_new = -w_deg × degradation_cost / 40.0`

相同满功率一小时充/放电事件在 group size 100 和 1841 下均得到 degradation cost = 40.0、normalized degradation reward = -1.0。BESS degradation 物理公式和 reward weight 未修改。

## 5. Supplemental 特征统一归一化

归一化只发生在 `IDCGridMultiAgentEnv` 的 observation/state 构造层。BESS actor observation 与 centralized state 接收同一数组，MLP critic 直接读取该 centralized state，HGTA GraphBuilder 只把同一数值映射到相应节点。

| 特征 | reference | 送入网络的值 |
|---|---:|---|
| SOC | 1.0 | `SOC / 1` |
| BESS energy | 10,000 kWh | `energy / 10000` |
| P_IDC | 25,000 kW | `P_IDC / 25000` |
| P_grid/site net power | 27,000 kW | `P_grid / 27000` |
| charge power | 2,000 kW | `charge / 2000` |
| discharge power | 2,000 kW | `discharge / 2000` |

27 MW grid reference 来自 25 MW facility rated power 加最大 2 MW BESS 充电功率。所有 reference 均由正式环境中的 facility/BESS 参数派生并写入 run/checkpoint metadata，未在多个实现点复制计算。

normal 24 小时轨迹的归一化统计：

| 特征 | min | mean | max |
|---|---:|---:|---:|
| SOC | 0.2684 | 0.3641 | 0.9000 |
| BESS energy | 0.2684 | 0.3641 | 0.9000 |
| P_IDC | 0.7355 | 0.7513 | 0.7686 |
| P_grid | 0.6074 | 0.6871 | 0.7568 |
| charge | 0.0000 | 0.0877 | 1.0000 |
| discharge | 0.0000 | 0.1250 | 1.0000 |

high 轨迹中 P_IDC 归一化范围为 0.8038–1.0001，P_grid 为 0.7442–0.9144。所有值有限、无 NaN/Inf，并处于预期 O(1) 量级。

## 6. MLP/HGTA 输入语义与结构不变性

HGTA schema/builder 版本更新为 `hgta_graph_v3_normalized_supplemental` / `hgta_builder_v3_normalized_supplemental`，用于显式标识输入语义变化。旧 `idc_power_ref_kw=2000` 和 `grid_power_ref_kw=4000` 的物理 metadata 更新为 25000/27000；GraphBuilder 不再使用这些 reference 对 state 做第二次除法。

MLP 与 HGTA 因而看到完全一致的 normalized centralized state。HGTA 的 IDC node 直接接收 `[P_IDC/25000, P_grid/27000]`，BESS node 直接接收 `[SOC, energy/10000, charge/2000, discharge/2000]`。

结构验收结果：

| 项目 | 结果 |
|---|---|
| HARL actor observations | `[288, 288]`，不变 |
| padded actions | `[22, 22]`，不变 |
| effective actions | IDC 22，BESS 1，虚拟 BESS 维仍为 21 |
| centralized state | `[364, 364]`，不变 |
| HGTA nodes | 39，类型/数量不变 |
| HGTA edges | 90，拓扑不变 |
| relation types | 14，不变 |
| HGTA hidden / heads / layers | 32 / 4 / 2，不变 |

## 7. 其他 scale-dependent reference 小范围审计

审计范围限定为与 `server_group_size`、`task_workload_scale`、IDC power 直接相关的 reference。

- `lambda_ref`、`queue_ref`、`queue_capacity_ref` 随 workload scale 同比例变化；arrival、completed work、backlog、urgent/overdue work 使用同一 work scale，归一化语义保持一致。
- `cost_ref`、carbon/cost 物理量和 `peak_power_ref_kW` 随 IDC power scale 变化，属于应随设施功率增长的 A 类量；只把 BESS degradation 从 IDC cost reference 中拆出。
- completed/backlog 回归示例在 group size 100/1841 下完全一致：`r_done=0.12370448`、`r_queue=-0.02020728`、`r_urgent=0`、`r_unused=0`。
- task forecast 仍按 `lambda_ref` 归一化；perfect/noisy/none 机制及 future truth 隔离测试通过。

未做 reward 重构，也未调整 reward weights、grid reward、branch thermal rating 或 Safe RL。

## 8. Checkpoint 与模型加载语义保护

输入维度虽然不变，物理语义已经改变，因此 checkpoint schema 升级为：

`idc-mappo-training-resume-v2-idc25-semantics`

新增并持久化：

- `input_semantics_version = idc25-normalized-supplemental-v1`
- `supplemental_normalization_version = physical-reference-v1`
- 六个 supplemental reference
- 正式 `IDC_SCALE_CONFIG` fingerprint

训练 resume compatibility、fixed evaluation environment fingerprint 和 model loader 都包含正式 IDC scale/语义。缺少新语义字段的旧约 1 MW resolved config/checkpoint 会被明确拒绝，不会静默加载到 25 MW 环境。

## 9. 测试与一次更新训练 smoke

### 9.1 定向/回归测试

回归命令覆盖 Phase 2.7、forecast isolation、多智能体 shapes/rollout/parity、HGTA builder/interface、checkpoint resume、既有 25 MW AC OPF 和动作可控性：

`40 passed, 15 skipped, 656 warnings in 126.05 s`

15 个 skip 是原有需要通过环境变量注入外部正式 run-dir 的产物门禁；本阶段要求的三个 run-dir 已由下面的实际一次更新训练独立生成并检查。warnings 均为 pandapower 对旧 net tap metadata 的 deprecation warning，不是数值或 OPF 失败。

### 9.2 三组 one-update smoke

每组均为 1 update、24 episode length、2 rollout threads、48 environment steps：

| 方法 | status | finite loss/value/entropy/logprob | actor/critic update | OPF/MEF | masking/HAPPO/HGTA |
|---|---|---|---|---|---|
| MAPPO + MLP | completed | pass | pass | 1.0 / 1.0 | BESS effective=1, padded=22；虚拟梯度=0 |
| HAPPO + MLP | completed | pass | 2 actors + critic 各 1 次 | 1.0 / 1.0 | factor nonfinite=0；虚拟 factor contribution=0 |
| HAPPO + HGTA | completed | pass | 2 actors + critic 各 1 次 | 1.0 / 1.0 | graph/embedding/attention nonfinite 均 0；critic parameter delta=0.1046 |

关键实际数值：MAPPO MLP value loss 77.9869；HAPPO MLP 52.2477；HAPPO HGTA 46.2374。三者的 policy loss、entropy、ratio、value prediction 和梯度范数均有限。HGTA 为 39 nodes、90 edges，critic gradient 存在；HAPPO importance factor 均为正且有限；dummy padded actions 没有贡献到物理 ratio、entropy 或 gradient。

首次 MAPPO run 在 rollout 前发现新 metadata 代码对两线程子进程 VecEnv 错误访问 `.envs[0]`。已改为从一次性单环境启动验证结果传递同一语义 reference；定向测试通过后重跑成功。失败 run 没有执行 update，未被当作验收结果。

## 10. 必答问题

**Q1. 25 MW 是否已经正式写入训练环境？**  是。正式 config、环境 metadata、训练 compatibility 和 evaluation fingerprint 均使用 25 MW。

**Q2. 最终 server_group_size / task_workload_scale 是多少？**  均为 **1841**；20 个 server groups 不变。

**Q3. 25 MW 正式额定定义是什么？**  最大合法 compute action=1、planned total load=0.65、30 °C 高温条件下，facility-level `P_IT + P_cooling + P_others ≈ 25 MW`。

**Q4. SLA reward scale 如何修复？**  用与 workload/MW 无关的 `sla_penalty_ref=50`，同一 task-count/priority/lateness 事件在 100 与 1841 尺度下得到相同 normalized penalty。

**Q5. BESS degradation reward scale 如何修复？**  使用 BESS 自身的 `2000 kW × 1 h × 0.02 = 40` reference，不再复用 IDC-scaled cost reference。

**Q6. supplemental power/energy features 如何归一化？**  依次用 1、10000 kWh、25000 kW、27000 kW、2000 kW、2000 kW，在 observation/state 构造层只归一化一次。

**Q7. MLP critic 与 HGTA 是否看到一致的物理信息和归一化语义？**  是。二者共享 normalized centralized state；HGTA 只做结构映射，没有二次 normalization。

**Q8. actor / state / topology / action dimensions 是否不变？**  是。actor obs 288/288、state 364、padded action 22/22；HGTA 39 nodes、90 edges、14 relations，架构 32/4/2。

**Q9. future leakage fix 是否仍通过？**  是。future truth isolation 以及 perfect/noisy/none forecast 回归通过，新 workload scale 下 forecast normalization 有限且语义不变。

**Q10. 三组 1-update smoke 是否全部通过？**  是。MAPPO+MLP、HAPPO+MLP、HAPPO+HGTA 均完成 1 update，数值有限，AC OPF/MEF success rate 均为 1.0。

**Q11. 是否还有阻止正式长训练的环境级 blocker？**  本阶段验收范围内没有。结论是 **READY FOR FORMAL TRAINING**。这不表示长训练一定收敛或某种算法更优；这些只能由后续正式实验回答。

## 11. Git 记录与交付边界

- HEAD before：`9957ebd` (`test: validate 25mw idc controllability`)
- git status before：已有用户修改 `.gitignore`、`configs/config_ultimate.py` 中 MEF cache bin、`env_wrappers/grid_coupled_env.py`、`grid_model/grid_cache.py`，以及多份未跟踪报告/测试文件。
- 本任务没有覆盖或混入上述用户改动；`configs/config_ultimate.py` 提交时仅选择本阶段 config/reward hunks。
- 本任务修改：正式 scale/reward config、IDC reward/env、multi-agent supplemental state、HGTA schema/builder、checkpoint/evaluation/model guard、训练 metadata、25 MW diagnostic 兼容和相关测试。
- 诊断输出：`outputs/idc_25mw_phase27/probe.json`（本地生成，不纳入提交）。
- 建议/实际 commit message：`fix: formalize 25mw idc scaling and normalization`
- commit SHA：报告生成时尚未创建提交；实际 SHA 在完成提交后记录于交付消息和仓库 `git log -1`。这是 Git commit 对包含自身报告文件的 SHA 无法自引用所致。

## 12. 最终结论

**READY FOR FORMAL TRAINING**

25 MW 已成为物理链贯通、奖励尺度可解释、MLP/HGTA 输入语义一致、旧 checkpoint 不会静默混用的正式 baseline。Safe RL 仍不进入本阶段主线；没有开始长训练，也没有作性能性结论。
