# 算法框架搭建——第十三部分：统一算法接口与 HAPPO＋MLP 接入

## A. HARL HAPPO 实现审计

审计源为 `C:\Users\bulio\Desktop\IDC\HARL`，HEAD `050ad6a294fe9f7572985dea910d59ea6d4f94b4`。核心调用链为 `OnPolicyHARunner.train` → `OnPolicyActorBuffer` → `HAPPO.train/update` → `VCritic.train`。

1. Agent 顺序：`fixed_order=true` 时为 `range(num_agents)`；否则每 update 调用 `torch.randperm(num_agents)`。
2. 当前正式配置为固定顺序 `[IDC, BESS]`。这是项目针对异构有效动作空间的显式选择，同时也是配置已验证值；不是 HARL 强制默认。
3. 随机顺序分支由 PyTorch 全局 RNG 控制。项目 checkpoint 保存/恢复 `torch_cpu` 和 `torch_cuda` RNG 状态，因此恢复后下一顺序连续一致。
4. factor shape 为 `[episode_length, n_rollout_threads, 1]`，本阶段为 `[24, 2, 1]`。
5. 每 update 起点重新初始化为全 1；factor 不跨 update 保留，因此 post-update checkpoint 不保存已消费 factor。
6. HARL 原生 factor 更新本身不乘 active mask；active mask 在 HAPPO surrogate loss 中使用。项目保持该语义。当前环境两 agent 全程 active。
7. 每个 agent 更新前在 rollout action 上计算旧策略 log-prob，更新后用刚更新策略计算新 log-prob。`share_param=false` 下，在轮到该 agent 前它未被其他 agent 修改，所以更新前策略就是 rollout 策略。
8. HARL 先按动作维聚合单 agent 的 `exp(new-old)`，再将标量比乘入累计 factor。项目将动作聚合限定到有效维：IDC 22 维、BESS 1 维；不会聚合 BESS 的 21 个 padded 虚拟维，也不是把两 agent 的 padded 动作拼成一次联合 44 维聚合。
9. HARL 实际顺序是：returns/GAE 已计算 → 按顺序更新 actor/factor → 最后仅调用一次 critic train。不是预设的 critic-first 顺序。
10. HARL `share_param=true` 构造同一个 actor 对象的多个引用；非共享分支构造独立 actor。当前 IDC/BESS 必须走 `share_param=false`，共享参数因有效动作 mask 不同而被项目明确拒绝。
11. 当前双 Agent 配置走非共享、固定顺序分支。
12. 直接复用：HARL actor 网络、HAPPO minibatch/update、rollout buffer、critic buffer、GAE、ValueNorm、VCritic loss、采样和 after-update 语义。
13. 项目 adapter：外部 VecEnv 注入、IDC/BESS 有效动作 mask、统一方法/critic 构造、factor 诊断、结构化日志、严格 checkpoint、固定评估来源信息。
14. 未修改 HARL 上游源码。

与 HARL 原生实现唯一有意的数学适配是：HAPPO ratio/factor 只在物理有效动作空间上计算。它等价于在真实动作空间 `[22, 1]` 上运行 HAPPO；padded 虚拟维不是环境动作，不应属于策略概率测度。Actor 参数 shape 仍兼容，但完整 checkpoint 通过方法/critic/mask 版本显式区分，不能冒充未适配的上游 HAPPO checkpoint。

准确 update 顺序：

```text
rollout
→ compute returns/GAE
→ determine agent order
→ IDC HAPPO update
→ factor *= IDC effective ratio
→ BESS HAPPO update（使用更新后的 factor）
→ factor *= BESS physical-dim ratio
→ critic update（一次）
→ logger
→ after_update
→ checkpoint
```

## B. 新增与修改文件

新增：

- `marl/methods/method_registry.py`、`marl/methods/__init__.py`
- `marl/critics/critic_factory.py`、`marl/critics/__init__.py`
- `marl/algorithms/update_strategy.py`
- `train/train_harl_short.py`
- `eval/eval_harl_fixed.py`
- `marl/tests/test_unified_algorithm_interface.py`
- `marl/tests/test_happo_mlp_update.py`
- 本报告

修改：

- `configs/harl_mappo_short.yaml`
- `marl/algorithms/__init__.py`
- `marl/runners/idc_mappo_runner.py`
- `marl/logging/training_metrics.py`
- `marl/checkpointing/training_checkpoint.py`
- `marl/evaluation/model_loader.py`
- `train/train_harl_mappo_short.py`
- `eval/eval_harl_mappo_fixed.py`

保持不动：

- HARL 上游仓库；
- 环境物理模型、Reward、observation/state/action 维度；
- BESS Bridge 物理切片；
- MAPPO loss/GAE；
- 学习率、clip、entropy、gamma、lambda；
- Grid Reward 与 Safe RL 开关；
- 固定评估场景内容。

工作区在本阶段开始前已经包含未提交的 Part 1–12 文件和修改；本阶段没有覆盖或回退这些用户改动。

## C. 统一方法与配置接口

显式组合及状态：

| algorithm | critic | method_id | 状态 |
|---|---|---|---|
| mappo | mlp | `MAPPO_MLP` | supported |
| happo | mlp | `HAPPO_MLP` | supported |
| happo | hgta | `HAPPO_HGTA` | registered, fail-fast |
| mappo | hgta | — | unsupported, fail-fast |

`HAPPO_HGTA` 的固定错误为：`HGTA critic interface registered but implementation is not available in Part 13`。

统一入口：

```powershell
python train/train_harl_short.py --algorithm mappo --critic-type mlp
python train/train_harl_short.py --algorithm happo --critic-type mlp
```

旧 `train/train_harl_mappo_short.py` 的导入和旧命令配置保持兼容；算法不再从文件名、类名或默认 actor 类型猜测。HAPPO 输出路径包含 `happo/idc_bess_happo_short_HAPPO_MLP`。resolved config、run metadata、checkpoint、evaluation manifest 均记录：

```text
algorithm_name
critic_type
method_id
algorithm_implementation_version
critic_interface_version
```

## D. Runner 和 Update Strategy

`IDCOnPolicyMARunner` 仍是唯一采样 runner，公共拥有 VecEnv、rollout、buffer、GAE、Bridge、日志、checkpoint 和 evaluation。算法差异只由：

- `MAPPOUpdateStrategy`：直接调用 HARL `OnPolicyMARunner.train`；
- `HAPPOUpdateStrategy`：复用 HARL 顺序更新结构，并调用项目 `EffectiveActionHAPPO`。

HAPPO 的 actor 训练结果在返回 logger 前按 canonical `[IDC, BESS]` 对齐；该操作只修正日志索引，不改变更新顺序或数学状态。

## E. Critic 统一接口

`MLPCentralizedCritic` 是 HARL `VCritic` 的透明 adapter，输入契约固定为 centralized state tensor `[batch, 294]`，提供：

```text
forward / get_values / evaluate
train / update
state_dict / load_state_dict / optimizer_state
to / train_mode / eval_mode
```

MLP 网络、optimizer 和 value loss 均未改变。`critic_type=hgta` 已注册接口身份，但在构造前 fail-fast，不生成假图、identity 或虚假输出。

## F. HAPPO 顺序更新和 Factor

默认顺序为 `IDC → BESS`。随机分支仍支持 `torch.randperm`，同 seed 单测复现；checkpoint 保存决定下一顺序的 PyTorch RNG。

唯一正式 smoke 的 factor：

| 阶段 | mean | min | max |
|---|---:|---:|---:|
| initial | 1.000000 | 1.000000 | 1.000000 |
| after IDC | 1.075187 | 0.724011 | 1.538155 |
| final | 1.071274 | 0.675560 | 1.502483 |

`factor_nonfinite_count=0`，factor 明确不是无条件恒为 1。两个 actor 各训练一次，critic 训练一次；IDC actor grad norm `10.930435`、BESS `2.830302`、critic `116.440865`，均 finite。

## G. BESS 有效动作 Mask

HAPPO 的 old/new log-prob、ratio、surrogate、clip、entropy、approximate KL、factor 更新及诊断均调用同一有效动作 mask。唯一 smoke 验收：

```text
bess_happo_factor_effective_physical_max_diff = 0
bess_happo_virtual_factor_contribution = 0
bess_virtual_mean_grad_norm = 0
bess_virtual_log_std_grad_norm = 0
bess_virtual_entropy_optimization_contribution = 0
```

IDC 仍使用 22 维；BESS padded 22 维但仅第 0 维有效。未强制虚拟动作输出为 0，未修改物理 Bridge。

## H. 日志与 Checkpoint

Step/Episode canonical 列未修改。Update 新增通用方法字段、顺序字段、HAPPO factor 全阶段统计和 BESS factor 诊断。MAPPO 的 HAPPO 专属列为空且 `happo_available=false`，不伪造为 0。

Checkpoint 记录算法、critic、method、实现版本、critic 接口版本、顺序策略、mask/shape 兼容信息、completed update 和 Python/NumPy/PyTorch RNG。factor 每 update 重置，因此不跨 update 保存。

新增 `--load-checkpoint-only`：严格恢复完整训练 checkpoint 后不执行 update。HAPPO smoke checkpoint load-only 成功恢复 `saved_update=1/global_step=48/episodes=2`。负向实测 `MAPPO_MLP` 加载 `HAPPO_MLP` checkpoint 时明确抛出 `CheckpointCompatibilityError`；单测另覆盖 MLP/HGTA 和 fixed/random order 拒绝。

## I. 固定评估兼容

Loader 根据 resolved config 的显式方法身份构造 MAPPO 或 HAPPO actor；不加载 critic、optimizer 或 HAPPO factor。Part 11 旧 MAPPO resolved config 通过有版本边界的 `MAPPO_MLP` legacy migration 继续加载。

HAPPO smoke actor 在 `smoke_v1/scenario_000_seed_02026` 完成唯一 24 步确定性评估：`status=completed`、`failure_count=0`、`critic_metrics_available=false`、actor gradient absent、actor parameter unchanged、eval mode true。evaluation manifest 记录 `HAPPO_MLP/mlp` 及两项实现版本。

## J. MAPPO 零回归

命令：

```powershell
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe train/train_harl_short.py `
  --algorithm mappo --critic-type mlp --seed 7110 --updates 1 `
  --episode-length 24 --rollout-threads 2 --device cpu `
  --output-dir runs/part13_mappo_zero_regression --checkpoint-interval 1
```

与 Part 12 `update_000001.pt`/首 update 逐项比较：

| 项目 | 最大差异 |
|---|---:|
| 48 行 Step 全公共字段 | 0 |
| 2 行 Episode 全公共字段 | 0 |
| Update 训练字段（排除时间/run_id） | 0 |
| rollout actions / log-probs | 0 / 0 |
| actor0 / actor1 参数 | 0 / 0 |
| actor0 / actor1 optimizer | 0 / 0 |
| critic 参数 / optimizer | 0 / 0 |
| critic rewards | 0 |
| effective-action diagnostics | 0 |

结论：统一接口重构没有改变 MAPPO 训练轨迹或状态。

## K. HAPPO 1-update Smoke

本阶段只执行一次有效 HAPPO 正式训练命令；此前没有 HAPPO update，之后仅做 read-only 审计/load-only/evaluation：

```powershell
C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe train/train_harl_short.py `
  --algorithm happo --critic-type mlp --seed 7110 --updates 1 `
  --episode-length 24 --rollout-threads 2 --device cpu `
  --output-dir runs/part13_happo_mlp_smoke --checkpoint-interval 1
```

结果：exit 0，约 `49.3 s`；48 Step rows、2 Episode rows、1 Update row、global_step 48、episodes 2、status completed。OPF/MEF 成功率 1.0，Reward reconstruction 最大误差 0，无 NaN/inf，checkpoint 严格保存并 load-only 成功。

固定评估命令使用 `eval/eval_harl_fixed.py`，单场景 24 步，exit 0，约 `70.5 s`。

产物：

- `runs/part13_happo_mlp_smoke/gym/idc_bess_padding/happo/idc_bess_happo_short_HAPPO_MLP/seed-07110-2026-07-29-12-09-07/`
- `evaluations/part13_happo_mlp_smoke/`

## L. 测试结果

| 命令/集合 | 结果 | 耗时 | warnings |
|---|---:|---:|---|
| `pytest -q test_unified_algorithm_interface.py test_happo_mlp_update.py` | 12 passed | 16.04 s | 0 |
| `pytest -q test_bess_effective_action_mask.py` | 12 passed | 112.72 s | 800 pandapower deprecation |
| `pytest -q test_training_metrics_logger.py` | 13 passed | 与入口组合 6.79 s | 16 pandapower deprecation（组合） |
| `pytest -q test_training_checkpoint_resume.py`（绑定 Part 9 产物） | 15 passed | 4.28 s | 0 |
| `pytest -q test_parallel_rollout_sampling.py`（绑定 Part 10 产物） | 18 passed；依赖读取失败项单独重跑 1 passed | 45.81 s + 18.28 s | pandapower deprecation；1 条 pytest cache 权限 warning |
| `pytest -q test_fixed_evaluation.py`（绑定 Part 11 产物） | 21 passed | 17.32 s | 48 pandapower deprecation |
| 正式入口快速结构 3 项 | 3 passed | 4.75 s | 1 条 pytest cache 权限 warning |

完整 `test_harl_mappo_formal_entry.py` 曾在 304.1 秒命令上限被终止，未产生失败断言；其包含的 1-update 生命周期随后由真实统一入口 MAPPO 命令独立完成并获得逐项零差异证据，因此未再次重复整文件物理回归。

警告均为 pandapower 对旧网络 `tap_dependency_table` 的弃用提示或 pytest cache/temp 沙箱权限；无训练数值、OPF、MEF、Reward 或算法 warning。

最终残留进程检查：`Get-Process python` 无结果；无残留 worker。

## M. 问题分类

### 阻塞进入 HAPPO 5-update

无。

### 第十四阶段必须监控

- factor 当前 finite，但首 update 范围约 `0.676–1.502`；继续监控 mean/min/max 和 nonfinite count。
- IDC 与 critic 首 update 的原始 grad norm 较大并触发既有 max-grad-norm 裁剪语义；继续监控，不在本阶段改超参数。
- 继续监控 BESS physical action 的变化强度和五项虚拟维零贡献不变量。

### HGTA 前必须完成

- 实现真正的 heterogeneous graph batch 类型及 HGTA critic；当前只允许 fail-fast。
- 为图输入补充与统一 critic batch 接口相配套的序列化和模型 shape 兼容测试。

### 建议改进

- 后续可把历史文件名 `idc_mappo_runner.py` 完全迁移为算法中性名称；当前类和实现已经统一，旧文件名仅为兼容入口，不影响语义。

## N. 第十三部分判断

| 判断项 | 结论 |
|---|---|
| 统一算法接口 | 通过 |
| MAPPO 零回归 | 通过，逐项最大差异 0 |
| HAPPO 更新语义 | 通过，符合审计后的 HARL 顺序语义 |
| HAPPO factor | 通过，finite、非恒 1、第二 agent 使用更新后 factor |
| BESS 有效动作 Mask | 通过，五项零贡献不变量精确为 0 |
| 日志兼容 | 通过，Step/Episode 不变，Update 显式区分方法 |
| Checkpoint 兼容 | 通过，strict load-only 与交叉拒绝成立 |
| 固定评估兼容 | 通过，HAPPO actor 单场景 24 步完成 |
| HGTA 接口预留 | 通过，接口稳定且未伪造实现 |

总体判断：**1. 通过，可以进入 HAPPO＋MLP 5-update Gate。**

## O. 下一阶段唯一建议

只进入一次受控的 HAPPO＋MLP 5-update Gate，保持本阶段全部环境、Reward、超参数和 Mask 不变，逐 update 监控 factor 范围/finite、两 actor 与 critic 梯度以及 BESS 五项零贡献不变量；不要同时引入 HGTA 或性能比较。
