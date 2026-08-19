# 第十八部分阻塞修复：价值网络初始化与策略动作采样的随机数隔离

日期：2026-07-29  
项目：`C:\Users\bulio\Desktop\IDC\ultimate_simplify`  
统一种子：`7110`；Critic 独立初始化种子：`207110`  
结论：随机数隔离修复及第十八部分硬性 Gate 全部通过；未运行长训练，未进行性能比较。

## A. 问题复现

修复前，三种方法的 Actor 初始参数、环境初始状态和 reset 输出一致；`MAPPO_MLP` 与 `HAPPO_MLP` 的首次 48 步轨迹一致，但 `HAPPO_HGTA` 的 2112 个动作元素均与 MLP 方法不同。差异发生在任何参数更新之前，因此不是 MAPPO/HAPPO 更新数学导致。

复现审计进一步确认：Actor 构造结束时三种方法的全局 Torch RNG 状态相同；构造 Critic 后，MLP 与 HGTA 因网络形状不同而消耗不同数量的全局随机数，使首次随机动作采样从不同 RNG 位置开始。

## B. 根因调用链

实际调用顺序为：

```text
train/train_harl_short.py
→ train/train_harl_mappo_short.py 配置解析与全局 set_seed
→ 环境/Bridge/Padding/并行环境构造
→ IDC Actor 构造
→ BESS Actor 构造
→ 统一 Critic factory（在此进入 MLP/HGTA 分支）
→ Critic 优化器构造
→ warmup/reset
→ HARL collect
→ Actor 随机动作采样
```

审计结论：Critic 网络参数初始化会消耗 Torch RNG；优化器、HGTA Graph Builder 和图结构 Schema 构造不消耗随机数。Critic 构造后到首次 Actor 采样前，没有发现其他按方法分支消耗 RNG 的操作。恢复训练虽会先构造临时 Critic，但现在也经过同一个隔离边界，随后由检查点覆盖模型、优化器、环境、Buffer 和真实训练 RNG 状态。

## C. 随机数隔离设计

采用“保存全局状态—独立初始化—无条件恢复”的统一方案：

1. Actor 保持原有顺序和种子正常初始化；
2. 保存 Python `random`、NumPy、Torch CPU，以及可用时全部 CUDA RNG 状态；
3. 使用固定规则 `critic_init_seed = base_seed + critic_init_seed_offset`；
4. 在独立种子下构造 MLP 或 HGTA Critic及其优化器；
5. 通过 `try/finally` 恢复进入作用域前的全部 RNG 状态；
6. Actor 后续随机采样继续沿用未被 Critic 污染的全局 RNG 流。

当前固定偏移为 `200000`，故种子 `7110` 对应 `critic_init_seed=207110`。偏移不随算法、Critic 类型或参数量变化。该设计不复制动作、不在每步重新设种子，也不关闭随机探索。

## D. 修改文件

生产代码和配置仅修改允许范围：

- `marl/utils/rng_isolation.py`：新增完整 RNG 捕获、恢复、隔离上下文和构造器；
- `marl/utils/__init__.py`：导出 RNG 工具；
- `marl/runners/idc_mappo_runner.py`：在统一 Critic 构造边界应用隔离；
- `train/train_harl_mappo_short.py`：解析、校验并传播 RNG 契约；
- `configs/harl_mappo_short.yaml`：加入固定种子规则；
- `marl/checkpointing/training_checkpoint.py`：写入最终及逐 update manifest；
- `marl/evaluation/model_loader.py`：加载和评估时校验兼容字段。

新增或更新的测试/只读审计脚本：

- `marl/tests/test_rng_isolation.py`；
- `marl/tests/audit_first_rollout_equivalence_part18.py`；
- `marl/tests/audit_resume_exactness_part18.py`；
- `marl/tests/audit_repeated_run_exactness_part18.py`；
- `marl/tests/audit_three_methods_part18.py`；
- 相关统一接口与正式入口测试。

未修改 Actor、Critic 数学结构、HGTA 图结构、环境、Bridge、Buffer、Reward、GAE、MAPPO/HAPPO、动作 Mask 或训练超参数。

## E. 隔离作用域实现

`marl/utils/rng_isolation.py` 同时处理四类状态：

- Python `random.getstate()/setstate()`；
- NumPy `get_state()/set_state()`；
- Torch CPU `get_rng_state()/set_rng_state()`；
- CUDA 可用时 `get_rng_state_all()/set_rng_state_all()`。

`isolated_global_rng(seed)` 使用上下文管理器和 `try/finally`，因此正常返回和构造异常都会恢复原状态；调用者无需手动清理。`build_with_isolated_rng()` 将完整 Critic 与优化器构造包在同一作用域。修复位于统一 Runner 边界，没有在 MLP 或 HGTA 内分别打补丁。

修复后，三种方法在 Actor 构造后及 Critic 构造后的 Torch RNG SHA-256 均为：

`ef959c82f5f2640c162eacb710139be4120994de557c60aaf315cfaee757daa9`

三种方法均记录 `critic_construction_changed_rng=false`。

## F. 配置和检查点元数据

配置、运行元数据、模型兼容信息、逐 update 和 final 检查点 manifest 统一记录：

```text
rng_isolation_version = critic-init-isolation-v1
critic_init_seed_rule = base_seed + critic_init_seed_offset
critic_init_seed_offset = 200000
critic_init_seed = 207110
actor_sampling_rng_isolated_from_critic_init = true
```

相同 base seed 下 MLP 与 HGTA 使用同一 Critic 独立种子。恢复时加载的是检查点保存的真实训练 RNG 状态；临时网络构造处于隔离作用域，不污染恢复状态。缺失或不匹配的 RNG 契约会进入检查点兼容性拒绝逻辑。

## G. 随机数单元测试

新增测试覆盖：

- MLP/HGTA 构造后 Python、NumPy、Torch CPU RNG 状态完全恢复；
- CUDA 可用时全部 CUDA RNG 状态恢复；
- 隔离构造不同 Critic 后，后续同形状 Torch 随机数 bit-exact；
- 同类型 Critic 在同一独立种子下所有参数可复现；
- 隔离作用域内故意抛异常后仍恢复全部状态；
- Actor 参数与优化器初始状态不受 Critic 类型影响；
- 优化器、Graph Builder 和 Schema 不额外污染 RNG。

最终完整测试结果：`125 passed, 41 skipped, 7 subtests passed`，0 failed，用时 433.56 秒。41 个 skip 为依赖历史产物的既有测试；本阶段要求的真实运行 Gate 已单独执行。警告主要来自既有 pandapower 弃用提示和受限环境下 pytest cache 提示。

## H. Actor初始化公平性

三种方法的 Actor 初始参数 bit-exact：

- IDC Actor SHA-256：`8db18f206e4745b9f051372a0f88d9e605cfdd89f97560d91770bd719b460a29`；
- BESS Actor SHA-256：`bafe67b84801897ddadc31d4d42fa5b0401f2219215b27818d8df0bf29ae1d7f`；
- 任意两方法 Actor 参数最大差异：`0`；
- Actor 优化器初始状态最大差异：`0`。

Critic 参数和结构允许不同，且没有被要求相同。

## I. 环境初始化公平性

统一 `seed=7110`、2 个 worker，实际 worker seeds 为 `[7110, 8110]`。三种方法的构造后环境状态、reset 输出、reset 后环境状态、局部观测和 294 维集中状态均 bit-exact，所有成对最大差异为 `0`。

环境、worker seed 规则、Bridge 和 Padding 均未修改。

## J. 第一次48步轨迹公平性

使用 `rollout_threads=2`、`episode_length=24`、CPU，不执行更新。结果文件为 `docs/dev_reports/PART18_FIRST_ROLLOUT_EQUIVALENCE_AFTER_RNG_FIX.json`。

结论：`all_three_methods_first_rollout_exact=true`。三种方法任意两两比较，下列最大差异全部为 `0`：

- IDC 22 维动作、BESS 补齐后的 22 维动作及 BESS 第 0 维物理动作；
- 动作 log-prob、Reward、done；
- 所有局部观测和 294 维集中状态；
- 任务状态、服务器负载、BESS SOC、`P_IDC`、`P_grid`；
- OPF/MEF、节点电压、线路负载及全部 info 投影；
- rollout 后环境状态。

每种方法得到 96 个唯一动作行，`actor_exploration_remains_stochastic=true`，说明一致性不是通过关闭探索或复制动作得到。MLP 与 HGTA 的 Critic value 最大差异 `0.8508996665477753`，属于明确允许的结构差异。

## K. 修复版HGTA连续5次更新

全新运行目录：`runs/part18_rng_fix_happo_hgta_5update_regression/.../seed-07110-2026-07-29-22-29-01`。

回归完成 240 steps、10 episodes、5 updates：

- 两个 Actor 与 HGTA Critic 均更新 5/5 次；
- HAPPO 顺序始终为 IDC→BESS；
- factor 每次从 1 初始化，始终 finite、positive，两处重算误差均为 0；
- BESS 第 1～21 维梯度、参数变化、熵贡献和 factor 贡献均为 0；
- 图结构保持 39 节点、90 有向边、7 类节点、14 类关系，三个图 hash 不变；
- 图输入、节点/关系/预测、动作、观测、状态和奖励无 NaN/Inf；
- OPF/MEF 成功率均为 1；Reward 重构误差和 shared reward 差异均为 0；
- update1～5 与 final 检查点完整，final 与 update5 的 9 个训练状态分区完全一致。

证据：`docs/dev_reports/part18_rng_fix_hgta_5update_audit.json`。

## L. update3→5精确恢复

从连续运行的 update3 恢复，在新目录完成 update4～5。`exact_resume_passed=true`：

- 最终模型、优化器、归一化器、RNG、环境、Runner、图结构及兼容信息逐项一致；
- 后 96 steps、4 episodes，以及 update4/5 的所有非计时字段差异数为 0、数值最大差异为 0；
- 仅 run id、绝对时间和 FPS 等预期运行标识/计时字段不同。

证据：`docs/dev_reports/part18_rng_fix_resume_exactness.json`。

## M. MLP短回归

`MAPPO_MLP` 和 `HAPPO_MLP` 均完成两次相同种子、全新的 1-update 运行。两组重复运行都满足：

- 48 steps、2 episodes、1 update；
- 模型、优化器、归一化器、RNG、环境、Runner 和兼容信息精确一致；
- step/episode/update CSV 的全部非计时字段精确一致；
- BESS 虚拟动作仍为零贡献；
- 仅 run id、时间和 FPS 等允许字段不同。

结果分别见 `part18_rng_fix_mappo_repeat_exactness.json` 和 `part18_rng_fix_happo_mlp_repeat_exactness.json`。本次按要求建立修复后的新公平性基准，没有把与修复前模型参数不同误判为回归。

## N. 检查点与固定评估

三种修复后方法均验证：

- final checkpoint 可读取且 manifest 哈希匹配；
- 同方法 load-only 成功，恢复到 update1 / global step 48 / 2 episodes，未执行额外更新；
- 3 个正确组合全部接受，6 个错误交叉组合全部以 `CheckpointCompatibilityError` 明确拒绝；
- `MAPPO_HGTA` 继续以 `ValueError` 明确拒绝；
- actor-only 固定评估完成 3 模型 × 1 固定场景 × 24 steps，共 72 steps、3 episodes，失败数 0；
- Actor 处于 eval 模式、无梯度、参数不变；BESS 仅第 0 维进入物理环境；
- 三种 manifest 均保留 RNG 隔离字段，HGTA manifest 继续保留图结构 hash。

固定评估目录：`evaluations/part18_rng_fix_three_methods_actor_only`。

## O. 第十八部分重跑结果

全部前置条件通过后，已完成原第十八部分剩余验收：三种方法各自全新 1-update、配置差异、日志格式、恢复与错误组合、固定评估、方法契约和教师摘要。

最终只读审计 `hard_fairness_gate_passed=true`，确认：

- 三种方法共用环境、数据、Actor、状态/观测、动作与 Mask、Reward、Buffer、GAE、日志、检查点和评估路径；
- 配置差异仅为算法实现、Critic 类型、method/experiment id 和输出目录等预期项；
- MAPPO 使用并行独立 Actor 更新语义，HAPPO 固定 IDC→BESS 顺序并使用 factor；
- HAPPO_HGTA 仅在预期的算法更新方式及集中式价值网络结构上不同；
- 日志行数均为 48 step、2 episode、1 update，非有限计数均为 0；
- 三方法首次 48 步轨迹完全一致。

证据：`docs/dev_reports/part18_rng_fix_three_method_final_audit.json`。

## P. 问题分类

原问题分类为“公平性阻塞 / 全局 RNG 耦合”，不是 Actor、环境、HGTA、HAPPO、MAPPO 或动作 Mask 的数学错误。修复将 Critic 初始化的随机数副作用封装在统一边界，不改变算法训练语义。

当前硬性阻塞项已清零。正式实验前保留以下监控项：

- 当前统一 logger 尚未逐 update 单独输出 value/return/advantage 的 min/max、attention score 与 weight 分项非有限统计、Actor/Critic 分项耗时；已有均值、标准差、总非有限计数和总更新时间足以支持本次短 Gate，但长训练建议补充监控；
- pandapower 仍产生既有弃用警告，应在依赖升级前持续观察；
- 41 个依赖历史产物的测试处于 skip，本阶段对应的真实运行 Gate 已独立执行并通过。

这些是可观测性和维护性事项，不影响本次随机数隔离、公平性、短训练、恢复或评估结论。

## Q. 最终判断

**选择 2：修复通过，第十八部分基本通过，但正式实验前有监控项。**

随机数隔离、首次 48 步公平性、HGTA 5-update、update3→5 精确恢复、MLP 重复性、三方法 1-update、检查点兼容性、load-only 和固定 Actor 评估全部通过。由于仍有上述非阻塞监控项，本报告不选择“完全通过”；也不据短流程结果做任何性能排序或优劣结论。

按任务要求，到此停止：未运行正式长训练，未调参，未进行性能比较。
