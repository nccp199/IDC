# 算法框架搭建——第十六部分：HGTA 集中式价值网络实现与 1-update 正确性 Gate

## A. 阶段目标和执行边界

本阶段在已经通过 `HAPPO_MLP` 的统一算法框架中，仅以 HGTA 替换集中式价值网络，并完成固定图、网络接口、静态测试、MLP 零回归、唯一一次 `HAPPO_HGTA` 1-update、完整 checkpoint、load-only 和 actor-only 固定评估 Gate。

严格保持环境、294 维状态、Bridge、Buffer、Actor、HAPPO 顺序更新、有效动作 Mask、Reward、GAE、OPF/MEF、学习率及其余训练超参数不变。没有引入 PyTorch Geometric/DGL，没有执行 HGTA 5-update、40-update或 MLP/HGTA 性能比较。

## B. 新增与修改文件

新增：

- `marl/graphs/__init__.py`
- `marl/graphs/graph_schema.py`
- `marl/graphs/graph_batch.py`
- `marl/graphs/graph_builder.py`
- `marl/critics/hgta_encoder.py`
- `marl/critics/hgta_critic.py`
- `marl/tests/conftest.py`
- `marl/tests/test_hgta_graph_builder.py`
- `marl/tests/test_hgta_encoder.py`
- `marl/tests/test_hgta_critic_interface.py`
- `marl/tests/test_happo_hgta_one_update.py`

修改：

- `configs/harl_mappo_short.yaml`：加入完整 `critic.hgta` 配置。
- `marl/critics/critic_factory.py`：正式构造 HGTA critic，保留原 MLP 路径。
- `marl/methods/method_registry.py`：注册已实现的 `HAPPO_HGTA`；`MAPPO_HGTA` 仍拒绝。
- `marl/runners/idc_mappo_runner.py`：向通用日志传递图元数据。
- `train/train_harl_mappo_short.py`：解析 critic 配置、构造统一 critic、写入恢复兼容性和 run metadata。
- `marl/checkpointing/training_checkpoint.py`：保存/验证图结构元数据。
- `marl/logging/training_metrics.py`：加入 HGTA 标识、规模和非有限值诊断。
- `marl/evaluation/model_loader.py`：actor-only manifest 从完整 checkpoint 继承训练 critic 类型和图 hash，但不构造 HGTA。
- `marl/tests/test_unified_algorithm_interface.py`：加入统一注册与交叉恢复拒绝覆盖。

本阶段没有修改禁止项。

## C. 固定图结构与特征 Schema

节点类型顺序固定为：

```text
global, idc, task_pool, server_group, bess, pv, bus
```

节点数依次为 `1, 1, 1, 20, 1, 1, 14`，总计 39。14 个有序关系按冻结顺序保存，共 90 条有向边：15 条线路双向 30 条、5 个变压器双向 10 条、ServerGroup↔IDC 40 条，及 TaskPool、IDC、BESS、PV、Global 各自的两条双向语义边共 10 条。IDC、BESS、PV 均接入 pandapower 内部母线索引 8（IEEE 编号 9）。

特征维度分别为 `global=13, idc=2, task_pool=10, server_group=6, bess=4, pv=3, bus=7`。未来预测保持独立 `state[136:280]` 共 144 维，不复制到节点。

固定归一化参考值为：IDC 2000 kW、Grid 4000 kW、BESS 10000 kWh、充/放电功率各 2000 kW；这些值来自项目固定系统容量并写入配置。母线参考值从冻结 IEEE-14 拓扑确定性派生：135 kV、94.2 MW、19 Mvar、母线索引 13。没有按 batch 动态归一化。

## D. 图构造器

`HGTAGraphBuilder` 只接受 finite 的 `[B,294]`。ServerGroup 将六段 feature-major 切片 `16/36/56/76/96/116 + i` 正确转置为 `[B,20,6]`；TaskPool、IDC、BESS、Global 和 144 维预测按冻结区间切片。

PV 当前小时由 `state[4:6]` 与 24 个标准正余弦编码逐点计算平方距离并取唯一最近点；24 个标准小时全部可精确恢复，且不访问环境内部未来对象。

边在每个样本内使用类型局部索引。批图为每种关系分别加源/目标类型偏移，不会把样本间节点串边；B=1、2、48 均验证通过。

## E. HGTA 关系注意力实现

实现为纯 PyTorch、32 隐藏维、4 头、每头 8 维、两层、不共享层参数、dropout=0、残差和 LayerNorm 开启。

对目标节点 `i`、源节点 `j`、关系 `r=(s,r,t)`、注意力头 `h`，实际实现为：

```text
q_i^h = W_Q,t^h h_i
k_j^h = W_K,s^h h_j
v_j^h = W_V,s^h h_j
k_j,r^h = R_K,r^h k_j^h
v_j,r^h = R_V,r^h v_j^h
e_ij,r^h = <q_i^h, k_j,r^h> / sqrt(8) + b_r^h
alpha_ij,r^h = softmax(e_ij,r^h over every incoming edge of target i, head h)
m_i = concat_h sum_(j,r -> i) alpha_ij,r^h v_j,r^h
h_i' = LayerNorm(h_i + activation(W_O,t m_i))
```

不同节点类型使用独立 Q/K/V/输出映射；每层、每个关系使用独立 `R_K`、`R_V` 和 bias。softmax 汇总同一目标节点的全部关系入边，按目标节点和 head 分别稳定归一化，不是全图 softmax。没有入边时消息为零，只保留残差路径。激活在输出映射后、残差和 LayerNorm 前。两层完全独立。

## F. 类型级汇总和未来预测融合

两层消息传递后，ServerGroup 20 节点和 Bus 14 节点分别做类型内均值，其余五种单节点类型直接保留。按冻结类型顺序拼接为 `7 × 32 = 224`。

未来预测独立通过 `144 → 64 → 32`，再与图摘要拼成 256 维，价值头为 `256 → 64 → 1`，输出严格为 `[B,1]`。没有对全部 39 节点做一次整体平均。

## G. 价值网络接口

`HGTACentralizedCritic` 提供与 MLP critic 一致的 `get_values`、`train/update`、`state_dict`、`load_state_dict`、`optimizer_state`、`to`、`train_mode`、`eval_mode`，并复用 HARL `VCritic` 已有 loss、mini-batch、优化器调用和梯度裁剪语义，仅替换其 value network。

正式组合为 `algorithm=happo, critic_type=hgta, method_id=HAPPO_HGTA`。`MAPPO_MLP` 和 `HAPPO_MLP` 路径保持不变；`MAPPO_HGTA` 仍明确拒绝。当前正式非循环配置正常；若启用 recurrent critic，HGTA v1 会 fail-fast，不会静默忽略 recurrent state/mask。

## H. 图结构版本、Hash 和 Checkpoint

版本：

```text
graph_schema_version       = hgta_graph_v1
graph_builder_version      = hgta_builder_v1
hgta_architecture_version  = hgta_critic_v1
```

冻结 hash：

```text
feature_schema_hash = 1944df31af4e68360173cabea4f47d2d13ca8167e576fdb64ea2f76b1ed82b23
topology_hash       = 0a85394f385693afc351ec03831fd01bc1b75f31019e4f68bbd2c7bf5115c9f1
graph_schema_hash   = 32a642d211c7421634a7e4e65afb54529232ce4da19f47ddfd9fdf52e9ae4b26
```

完整 checkpoint 保存 critic/method、三个版本、节点/关系顺序、特征顺序和维度、拓扑、归一化参考值、HGTA 配置及三个 hash；恢复兼容性逐层比较这些字段。

同构 `HAPPO_HGTA → HAPPO_HGTA` load-only 成功，恢复到 update 1、global step 48、episode 2。交叉 `HAPPO_HGTA → HAPPO_MLP` 在任何 update 前以 `CheckpointCompatibilityError` 明确拒绝，同时报告两侧 method/critic 差异。

## I. 静态与单元测试

覆盖内容包括：

- 294 维切片、ServerGroup 转置、TaskPool/IDC/BESS/forecast/PV 当前小时。
- 错误 shape、NaN、inf fail-fast。
- 39 节点、90 边、14 个关系、线路/变压器不遗漏、确定顺序和 hash 重建一致。
- B=1、2、48 图批和前向，样本隔离。
- `[B,294] → [B,1]`、finite attention/embedding/value、CPU、train/eval、固定种子复现。
- ServerGroup、BESS、当前价格、母线静态负荷、未来价格的信息敏感性。
- 合成反向传播覆盖类型投影、两层关系注意力、关系参数、forecast encoder、类型汇总和值头，所有已有梯度 finite。
- 统一 critic 外部接口、严格 state/optimizer load、recurrent 拒绝、方法注册。
- 正式 1-update 产物行数、更新计数、诊断和 checkpoint 元数据。

最终 HGTA 核心组结果为 `27 passed in 11.76s`，0 warning。

## J. MLP 零回归

分别重新执行 seed 7110、2 workers、24 steps、1 update 的 `MAPPO_MLP` 与 `HAPPO_MLP`，并与 Part 13 同 seed 基准比较。

两种方法均满足：

- `model_state` 42 个 tensor 最大差异 0。
- `optimizer_state` 114 个 tensor 最大差异 0。
- `normalizer_state`、Torch/Python/NumPy RNG、环境状态逐项精确相等。
- Step 48 行、Episode 2 行、Update 1 行；排除运行 ID/计时及本阶段新增纯诊断字段后，共同训练字段最大差异 0。
- checkpoint 的 `runner_state` 仅新增稳定性计数器字段；verification 仅新增审计/诊断字段，不改变训练数值状态。

MAPPO 新 run：

```text
runs/part16_mappo_mlp_zero_regression/gym/idc_bess_padding/mappo/idc_bess_mappo_short/seed-07110-2026-07-29-18-27-10
```

HAPPO 新 run：

```text
runs/part16_happo_mlp_zero_regression/gym/idc_bess_padding/happo/idc_bess_happo_short_HAPPO_MLP/seed-07110-2026-07-29-18-28-10
```

因此 HGTA 接入没有改变既有 MLP 数学或 HAPPO 更新语义。

## K. HAPPO＋HGTA 1-update

所有静态测试通过后，只执行了一次正式 HGTA 训练：

```text
algorithm=happo, critic_type=hgta, seed=7110
rollout_threads=2, episode_length=24, updates=1, device=cpu
checkpoint_interval=1
```

结果：exit 0，`status=completed`，Step 48、Episode 2、Update 1、global step 48。两 Actor 各更新 1 次，HGTA critic 更新 1 次；critic 参数变化范数 `0.1017950295`，优化器 step delta=1。

所有 HGTA 非有限计数为 0；HAPPO factor finite、positive，重建差异 0；BESS 21 个虚拟维的梯度/熵/factor 贡献仍为 0；共享 Reward 重建差异 0；OPF/MEF 成功率均为 1；动作范围检查通过；final checkpoint 保存成功。

本次 rollout step reward mean 为 `-0.9383148932`，仅作运行记录，不能据此判断性能。

正式 run：

```text
runs/part16_happo_hgta_smoke/gym/idc_bess_padding/happo/idc_bess_happo_short_HAPPO_HGTA/seed-07110-2026-07-29-18-33-24
```

## L. HGTA 参数和梯度审计

参数规模：

```text
HGTA critic = 102,673
MLP critic  = 11,245
ratio       = 9.1305469
```

从正式 final checkpoint 的 Adam `exp_avg` 审计首次 update，不执行第二次训练：

| 模块 | 参数 tensor | 参数数 | 非零 exp_avg tensor | exp_avg L1 | finite / step |
|---|---:|---:|---:|---:|---|
| 类型投影 | 14 | 1,664 | 14 | 2.312916 | 是 / 1 |
| 关系层 0 | 91 | 36,568 | 76 | 3.778487 | 是 / 1 |
| 关系层 1 | 91 | 36,568 | 76 | 3.284693 | 是 / 1 |
| Forecast encoder | 4 | 11,360 | 4 | 0.882116 | 是 / 1 |
| Value head | 4 | 16,513 | 4 | 58.158179 | 是 / 1 |

两层各 14 个关系的关系专属参数组均有非零 `exp_avg`；层 0 最小关系组 L1 为 `0.00190305`，层 1 为 `0.00124491`。

每层 15 个零 score 参数来自五类“每个目标节点恰好只有一条入边”的关系：IDC→ServerGroup、IDC→TaskPool、Bus→BESS、Bus→PV、IDC→Global。其 per-target softmax 恒为 1，因此 query、relation-key 和 score bias 数学上不可辨识；对应 relation-value 参数均非零，消息确实传递。多入边关系的注意力 score 参数非零。这是冻结图结构原因，不是关系被忽略。

正式日志中的 critic pre-clip/post-clip gradient norm 为 `53.461666 / 10.000003`，均 finite；单次 update 计算段为 `0.096377s`，该 run 内部 rollout+update wall metric 为 `72.356365s`，外部完整进程墙钟约 `225.4s`（包括导入、固定拓扑/环境构造、保存和退出）。

正式训练未预先安装进程峰值采样器，因此不伪造其 peak。作为可获得的独立内存指标，纯合成 B=48 HGTA 前向+反向、无 optimizer step 的 Windows `PeakWorkingSetSize` 为 `332.758 MiB`；HGTA 参数 tensor 本体为 410,692 bytes，final checkpoint 为 2,007,365 bytes。该指标只作资源记录，不作性能结论。

## M. 固定评估兼容

只评估 `smoke_v1 / scenario_000_seed_02026 / 24 steps`，从 HGTA final checkpoint 仅加载两个 actor。评估没有构造 HGTA、没有加载 critic optimizer、没有恢复图注意力参数。

结果：exit 0，`status=completed`，failure count 0，`critic_metrics_available=false`，actor parameters unchanged、actor gradients absent、actors in eval mode 均为 true；BESS 仍只执行物理第 0 维。manifest 记录 `HAPPO_HGTA`、`critic_type=hgta` 及完整 graph/topology/feature hash。

输出：

```text
evaluations/part16_happo_hgta_actor_only
```

## N. 测试结果与真实命令

主要正式命令：

```powershell
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe train\train_harl_short.py `
  --algorithm happo --critic-type hgta --seed 7110 --updates 1 `
  --episode-length 24 --rollout-threads 2 --device cpu `
  --output-dir runs\part16_happo_hgta_smoke --checkpoint-interval 1 `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path C:\Users\bulio\Desktop\IDC\ultimate_simplify\.tmp_harl_runtime
```

exit 0，外部墙钟约 225.4s，产物路径见 K。

同构 load-only：

```powershell
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe train\train_harl_short.py `
  --algorithm happo --critic-type hgta --seed 7110 --updates 1 `
  --episode-length 24 --rollout-threads 2 --device cpu `
  --output-dir runs\part16_hgta_load_only --checkpoint-interval 1 `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path C:\Users\bulio\Desktop\IDC\ultimate_simplify\.tmp_harl_runtime `
  --resume-checkpoint <PART16_HGTA_FINAL_PT> --load-checkpoint-only
```

exit 0，186.7s，输出 `status=checkpoint_loaded`；没有 update。

交叉拒绝使用同一命令但 `--critic-type mlp`，exit 1，92.1s，按预期抛出明确的 `CheckpointCompatibilityError`；没有 update。

actor-only 固定评估：

```powershell
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe eval\eval_harl_fixed.py `
  --suite-id smoke_v1 --scenario-id scenario_000_seed_02026 `
  --checkpoint <PART16_HGTA_FINAL_PT> `
  --output-dir evaluations\part16_happo_hgta_actor_only `
  --evaluation-id part16-happo-hgta-actor-only --device cpu `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path C:\Users\bulio\Desktop\IDC\ultimate_simplify\.tmp_harl_runtime
```

首次在模型加载阶段因新增 `copy.deepcopy` 漏导入 `copy` 而 exit 1，未建环境、未执行 step、未产生输出目录；补充导入后同命令 exit 0，34.3s（实际 evaluation 27.16s）。

最终测试分组：

```text
HGTA core + formal artifact:
27 passed in 11.76s, exit 0, 0 warning

HAPPO/MLP + checkpoint + logger + BESS mask:
29 passed, 15 skipped, 816 warnings in 115.21s, exit 0

fixed evaluation:
3 passed, 18 skipped, 16 warnings in 5.80s, exit 0
```

`skip` 均为未设置历史/外部产物路径的 artifact-gated 测试；warning 均为 pandapower 3.x 对旧 transformer spline characteristic 的既有 deprecation warning，不是 HGTA 数值 warning。

曾尝试将三组一次执行，外层 240s timeout 后发现其子 pytest 仍存活；该次不计作通过或失败。精确终止该次遗留 PID 后分组重跑全部得到上述有效结果。最终 `Get-Process python` 返回空，无残留训练、评估或测试进程。

## O. 问题分类

### 阻挡进入 HGTA 5-update

无。

### 下一阶段监控项

- HGTA 参数量约为 MLP 的 9.13 倍。
- critic 首次梯度触发既有 norm=10 裁剪；虽 finite，后续应逐 update 观察。
- 五类单入边目标的 attention score 参数结构性零梯度；relation-value 路径有效。后续解释注意力时不能把恒为 1 的单入边权重当作学习出的选择。
- 固定拓扑/环境初始化占完整进程墙钟的大部分；需要区分初始化、rollout 与 update 时间。
- value loss、节点类型摘要范数和关系注意力集中度仅作后续稳定性监控，不在 1-update 阶段调参。

### 后续实验解决

是否优于 MLP、参数匹配比较、特征/关系消融、跨电网规模泛化和注意力解释性，均不由本阶段回答。

## P. 第十六部分判断

| 判断项 | 结论 |
|---|---|
| 图结构正确性 | 通过：39 节点、90 有向边、14 关系 |
| 294 维状态重建 | 通过 |
| 节点特征归一化 | 通过：固定参考值、无 batch 动态归一化 |
| 批处理隔离 | 通过：B=1/2/48、无样本串边 |
| 关系注意力 | 通过：类型与关系专属参数、per-target/head softmax |
| 类型级汇总 | 通过：224 维，未整体平均 39 节点 |
| 未来预测融合 | 通过：144→64→32，最终 256→64→1 |
| 价值网络接口 | 通过 |
| HGTA 梯度 | 通过：主要模块 finite 且发生非零更新 |
| HAPPO 接入 | 通过：两 actor + critic 各一次更新 |
| BESS 有效动作 Mask | 通过：21 个虚拟维贡献保持 0 |
| MLP 零回归 | 通过：训练状态精确零差异 |
| Checkpoint | 通过：同构加载成功、交叉恢复拒绝 |
| 固定评估 | 通过：actor-only、参数不变、无梯度、hash 完整 |

总体选择：**1. 通过，可以进入 HAPPO＋HGTA 连续 5-update Gate。**

该结论仅表示实现与单次更新正确，不表示 HGTA 性能优于 MLP。

## Q. 下一阶段唯一建议

只进入一次受控的 `HAPPO_HGTA` 连续 5-update 稳定性 Gate，保持本阶段环境、seed 规则、两 worker、24 步 episode、Actor/HAPPO/Reward/GAE/Mask 和所有训练超参数不变；逐 update 监控 HGTA 非有限计数、critic 梯度裁剪前后、模块参数变化、HAPPO factor、BESS 虚拟维零贡献、OPF/MEF、checkpoint 与 fixed actor-only evaluation。不要同时引入性能比较、图消融或调参。
