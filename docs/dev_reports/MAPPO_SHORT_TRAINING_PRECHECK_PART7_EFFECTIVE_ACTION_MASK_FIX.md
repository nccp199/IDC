# MAPPO短训练前检查——第七部分：BESS有效动作Mask修复与重新验收

## 结论摘要

本次修复通过。BESS仍保持22维Actor输出、22维逐维`log_prob`和22维buffer动作，但MAPPO/HAPPO的ratio、clip、policy loss、entropy、KL与梯度只使用真实第0维；IDC仍使用全部22维。没有修改物理环境、Bridge、Action Adapter、reward、GAE、centralized critic、OPF/MEF/cache或动作输出shape。

第七部分重新判断为：**1. 通过，虚拟维不再进入MAPPO/HAPPO优化。**

## 1. 修改文件

### 新增

- `marl/algorithms/__init__.py`：导出项目侧有效动作算法适配器与公式辅助函数。
- `marl/algorithms/effective_action_mask.py`：实现`EffectiveActionMAPPO`、`EffectiveActionHAPPO`、masked ratio/entropy和一次update诊断。
- `marl/tests/test_bess_effective_action_mask.py`：覆盖Mask定义、MAPPO/HAPPO语义、梯度、诊断、checkpoint兼容和24步物理不变性。
- `docs/dev_reports/MAPPO_SHORT_TRAINING_PRECHECK_PART7_EFFECTIVE_ACTION_MASK_FIX.md`：本报告。

### 修改

- `marl/specs/agent_specs.py`：定义`PADDED_ACTION_DIMS={idc:22,bess:22}`、`EFFECTIVE_ACTION_DIMS={idc:22,bess:1}`和`effective_action_mask()`。
- `marl/specs/__init__.py`：导出上述Agent动作规格。
- `marl/runners/idc_mappo_runner.py`：正式runner按Agent构造项目侧masked actor；校验Agent顺序、动作维度及`share_param=false`。
- `marl/diagnostics/bess_policy_diagnostics.py`：诊断中的effective ratio、entropy、KL、clip fraction改用同一显式Mask，同时保留full未Mask对照量。
- `train/train_harl_mappo_short.py`：正式入口记录Mask元数据与真实update诊断。
- `marl/tests/test_harl_mappo_formal_entry.py`：正式入口测试增加Mask元数据、BESS虚拟梯度和ratio一致性断言。

### 保持不动

- `C:\Users\bulio\Desktop\IDC\HARL`：本次未修改HARL仓库，工作树检查为空；当前HEAD为`050ad6a294fe9f7572985dea910d59ea6d4f94b4`，其上游基线记录为`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`。
- `marl/bridges/harl_padded_bridge.py`及其他Bridge：动作切片仍只把BESS第0维传入物理环境。
- `marl/adapters/action_adapter.py`：未修改动作映射与有界动作实现。
- 物理环境、reward、GAE、critic、OPF、MEF、cache、VecEnv、buffer shape、IDC动作语义和训练超参数均未修改。
- BESS observation padding与LayerNorm问题保持现状，留作独立议题；本次只确认无NaN/inf。

说明：工作树中还存在此前阶段的修改和未跟踪文件；本次没有覆盖或清理这些既有改动。

## 2. Mask数据流

```text
marl/specs/agent_specs.py
  EFFECTIVE_ACTION_DIMS = {idc: 22, bess: 1}
  effective_action_mask(idc)  -> [1 x 22]
  effective_action_mask(bess) -> [1, 0 x 21]
        |
        v
marl/runners/idc_mappo_runner.py :: IDCOnPolicyMARunner.__init__
  校验Agent顺序(idc,bess)、padded action dim=(22,22)、share_param=false
  分别构造actor 0 effective_action_dim=22、actor 1 effective_action_dim=1
        |
        v
marl/algorithms/effective_action_mask.py
  EffectiveActionMAPPO / EffectiveActionHAPPO
  Mask在actor算法实例初始化时一次性建立于目标device
        |
        v
HARL actor.evaluate_actions(...)
  仍返回(batch,22)逐维bounded Box log_prob
        |
        v
effective_ratio / effective_entropy
  IDC聚合22维；BESS仅聚合第0维
        |
        v
PPO/HAPPO ratio -> clip -> surrogate -> policy loss -> backward
        |
        v
train/train_harl_mappo_short.py
  保存Mask规格与最后一次真实update诊断
```

Mask不是从动作值推断。它是Agent规格的一部分，初始化为持久的device tensor，并按输入`log_prob`的dtype/device进行广播兼容；没有每step从Python list重复构造。

## 3. 数学实现

设逐维新旧log probability为`new_j`和`old_j`，Agent Mask为`m_j`。

### Effective log ratio

```text
effective_log_ratio = sum_j m_j * (new_j - old_j)
```

对应实现：

```python
effective_log_ratio = (
    (new_log_probs - old_log_probs) * mask
).sum(dim=-1, keepdim=True)
```

### Effective ratio

```text
effective_ratio = exp(effective_log_ratio)
```

没有使用错误的`prod(exp(delta) * mask)`形式，因此Mask为0的维度不会把乘积变成0。

### Effective entropy

当前bounded affine-tanh分布集成使用同一采样动作的逐维log probability估计entropy：

```text
effective_entropy = -sum_j m_j * log_prob_j
```

随后按HARL已有`active_masks`语义聚合。IDC的Mask全为1，行为等价于原22维聚合；BESS只有第0维贡献entropy。

### PPO/HAPPO loss

`effective_ratio`先参与PPO clip和surrogate。HAPPO的`factor_batch`在masked surrogate形成之后相乘，因此HAPPO factor不会重新引入21个虚拟维的ratio。

## 4. MAPPO与HAPPO覆盖

| 算法 | 适配类 | Mask应用层 | 覆盖状态 |
| --- | --- | --- | --- |
| MAPPO | `EffectiveActionMAPPO` | `update()`调用共享`_masked_update()`，ratio、entropy、clip、loss及诊断统一使用Mask | 已接入正式`IDCOnPolicyMARunner`并完成正式1-update |
| HAPPO | `EffectiveActionHAPPO` | `update()`把`factor_batch`传入同一`_masked_update()`；factor应用于masked surrogate | 源码适配及直接update单测通过；未扩展为正式HAPPO runner，符合本阶段范围 |
| 未提供Mask的Box任务 | 两个适配类的`effective_action_dim=None` | 默认`effective_dim=padded_dim`，即全维有效 | 向后兼容单测通过 |

正式MAPPO入口不依赖pytest monkeypatch；`IDCOnPolicyMARunner`从项目算法registry自动构造两个masked actor。HARL上游源文件没有被本次修改。

## 5. 测试结果

固定环境：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:HARL_SOURCE_PATH='C:\Users\bulio\Desktop\IDC\HARL'
$env:PYTHONPATH=((Get-Location).Path + ';' + $env:HARL_SOURCE_PATH)
```

| 验证 | 完整pytest参数 | 结果 | warnings | 耗时 |
| --- | --- | ---: | ---: | ---: |
| Mask/ratio/entropy/MAPPO梯度/trunk/HAPPO/诊断/checkpoint单测 | `-m pytest marl/tests/test_bess_effective_action_mask.py -q -k 'not four_virtual_action_patterns'` | 11 passed，0 failed，1 deselected | 0 | 4.23s |
| 四种虚拟动作完整24步物理回归 | `-m pytest marl/tests/test_bess_effective_action_mask.py -q -k 'four_virtual_action_patterns'` | 1 passed，0 failed，11 deselected | 800 | 115.55s |
| 正式入口回归 | `-m pytest marl/tests/test_harl_mappo_formal_entry.py -q` | 9 passed，0 failed | 257 | 53.24s |
| 既有MAPPO smoke | `-m pytest marl/tests/test_harl_mappo_smoke.py -q -s` | 1 passed，0 failed | 201 | 47.02s |
| Grid/cache正式入口回归 | `-m pytest marl/tests/test_grid_cache_formal_entry.py -q` | 11 passed，0 failed | 56 | 71.26s |
| Bridge/padding/BESS monitor组合回归 | 见下方命令 | 12 passed，0 failed | 600 | 89.02s |

Bridge/padding组合命令：

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" -m pytest `
  marl/tests/test_harl_bridge_shapes.py `
  marl/tests/test_harl_bridge_parity.py `
  marl/tests/test_harl_bridge_rollout.py `
  marl/tests/test_harl_padding_shapes.py `
  marl/tests/test_harl_padding_observation.py `
  marl/tests/test_harl_padding_action_invariance.py `
  marl/tests/test_harl_padding_parity.py `
  marl/tests/test_bess_virtual_action_monitor.py -q
```

800条物理回归warning均为pandapower既有`tap_dependency_table`弃用警告，不是数值、动作或Mask异常。

补充说明：最终快速单测第一次补跑时只设置了项目根目录`PYTHONPATH`，在collection阶段以`ModuleNotFoundError: harl`退出，未执行测试逻辑。补齐固定`HARL_SOURCE_PATH`后同一测试集为11 passed。该环境性失败未通过改源码处理。

## 6. 修复前后对比

正式1-update命令：

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  train/train_harl_mappo_short.py `
  --config configs/harl_mappo_short.yaml `
  --harl-source C:\Users\bulio\Desktop\IDC\HARL `
  --runtime-path .tmp_harl_runtime `
  --seed 7110 `
  --updates 1 `
  --episode-length 24 `
  --rollout-threads 1 `
  --output-dir runs/part7_mask_acceptance `
  --scenario idc_bess_padding `
  --device cpu
```

命令退出码为0，完成1次update和24个环境步，保存两个actor和centralized critic；`nan_or_inf_detected=false`，动作范围检查通过。

| BESS指标 | 修复前 | 修复后正式1-update |
| --- | ---: | ---: |
| virtual entropy优化贡献占比 | 93.8% | 0% |
| virtual mean rows梯度norm | 15.5006 | 0.0 |
| virtual mean rows最大绝对梯度 | 非零 | 0.0 |
| virtual `log_std`梯度norm/max | 非零 | 0.0 / 0.0 |
| effective ratio mean | — | 0.9984898567 |
| effective ratio std | 0.22242（旧22维联合值） | 0.0319100432 |
| effective ratio min/max | — | 0.9264440536 / 1.0694487095 |
| physical第0维ratio mean/std | — | 0.9984898567 / 0.0319100432 |
| effective与physical ratio最大差异 | — | 0.0 |
| clip fraction | — | 0.0 |
| clip disagreement | 37.5% | 0.0 |
| approximate KL | 0.0248288（旧22维联合值） | 0.0005127774，与physical相同 |
| effective entropy | — | -0.5508778691 |
| physical mean row梯度norm | — | 1.7883851528 |
| physical `log_std`梯度绝对值 | — | 0.0655261725 |
| shared trunk梯度norm | 0.4922346（旧虚拟维干扰） | 0.0695433170，与physical-only计算严格一致 |
| total actor梯度norm（clip前） | — | 1.7909358848 |
| finite检查 | — | 通过 |

IDC同一次update仍使用22维全1 Mask：ratio mean/std为`0.9368818402 / 0.1834502220`，clip fraction为`0.2916666567`，KL为`0.0192499589`，shared trunk梯度norm为`0.5058286190`，总actor梯度norm为`13.6096104493`，所有诊断均finite。BESS Mask没有改变IDC聚合语义。

诊断与模型路径：

```text
runs/part7_mask_acceptance/gym/idc_bess_padding/mappo/idc_bess_mappo_short/seed-07110-2026-07-28-18-21-51/run_metadata.json
runs/part7_mask_acceptance/gym/idc_bess_padding/mappo/idc_bess_mappo_short/seed-07110-2026-07-28-18-21-51/models/
```

## 7. 物理回归

物理回归使用四个独立、相同seed的正式环境，IDC动作与BESS真实第0维固定相同，BESS虚拟21维分别采用：

1. 全0；
2. 全1；
3. 固定随机向量；
4. 每步变化的随机向量。

四个环境完成完整24步。测试递归比较每一步完整返回结构中的数组、字典、tuple/list、标量和布尔值，使用严格相等而不是近似容差；所有物理、reward、SOC、功率、OPF/MEF/cache相关可见字段差异均为0。由此确认虚拟维仍不进入Bridge后的物理链。

## 8. Checkpoint兼容

- IDC Actor输出维度仍为22。
- BESS Actor输出维度仍为22。
- `fc_mean.weight` shape仍为`[22, 32]`，`fc_mean.bias`为`[22]`，`log_std`为`[22]`，bounded Box的low/high均为`[22]`。
- Actor、buffer action和逐维log probability shape均未改变。
- Mask是算法适配器运行时从Agent规格重建的tensor，不改变Actor `state_dict`键或参数shape。
- 单测将原22维HARL MAPPO Actor的`state_dict`以`strict=True`加载到masked actor，`missing_keys=[]`且`unexpected_keys=[]`。
- 因此现有`padding-v1 / output-22`模型参数结构保持可加载；恢复正式训练时仍须由正式runner按Agent重新建立Mask语义。

## 9. 第七部分重新判断

**选择1：通过，虚拟维不再进入MAPPO/HAPPO优化。**

判断依据：

- BESS ratio、clip、policy loss、entropy和KL只使用第0维；
- BESS虚拟mean rows与虚拟`log_std`梯度严格为0；
- shared trunk梯度与physical-only反事实计算严格一致；
- MAPPO正式1-update通过且无NaN/inf；
- HAPPO共享同一masked update语义，直接update单测通过；
- IDC仍完整聚合22维；
- Actor/buffer/checkpoint仍保持22维；
- 四种虚拟动作模式的完整24步物理回归严格一致；
- 既有正式入口、smoke、grid/cache、Bridge和padding回归全部通过。

本报告到此停止，不进入第八部分，也未运行5-update或40-update训练。
