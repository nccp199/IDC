# MAPPO短训练前检查——第十二部分：5-Update短训练门槛与固定评估验收

## 结论摘要

第十二部分完成。唯一一次正式5-update训练从头到尾正常结束，准确产生240 transitions、10个完整episode和5条update日志；两个Actor与centralized Critic均发生有限且非零更新。BESS有效动作Mask在5个update中全部成立，虚拟21维的mean/log-std梯度、entropy贡献和参数变化均为0，真实第0维与shared trunk持续获得非零梯度。

Update 1、3、5均使用原冻结`smoke_v1`完成3场景确定性评估，共216 transitions、9 episodes、0失败。Update 5×场景2026的额外24步重复结果逐字段完全一致，最大差异0。Checkpoint之间Actor参数、动作轨迹和物理轨迹均发生有限变化。

未发现NaN/inf、动作越界、Mask失效、OPF/MEF失败、日志计数损坏、checkpoint损坏、死锁或残留worker。需要在40-update阶段重点监控Critic持续触发梯度裁剪、IDC部分update触发裁剪，以及BESS确定性策略逐渐偏向小幅持续充电。因此门槛结论为：

```text
总体判断 = 2. 基本通过，可以进入40-update但需重点监控
gate_decision = pass_with_monitoring
```

这只是工程门槛，不代表MAPPO已经收敛或优于任何基线。

## A. 冻结配置与执行命令

### A.1 最终生效配置

| 项目 | 最终值 | 核验结果 |
| --- | ---: | --- |
| algorithm | `mappo` | 通过 |
| seed | 7110 | 通过 |
| worker seeds | `[7110, 8110]` | 通过 |
| n_rollout_threads | 2 | 通过 |
| episode_length | 24 | 通过 |
| updates | 5 | 通过 |
| transitions/update | 48 | 通过 |
| num_env_steps | 240 | 通过 |
| share_param | false | 通过 |
| action_aggregation | `prod` | 通过 |
| bounded Box | true | 通过 |
| BESS effective/padded/virtual dim | 1 / 22 / 21 | 通过 |
| Grid Reward / Safe RL | false / false | 通过 |
| use_linear_lr_decay | false | 通过 |
| device | CPU | 通过 |
| torch / BLAS threads | 1 / 1 | 通过 |
| checkpoint | enabled、每update保存、final保存 | 通过 |
| keep_last | 5 | 通过 |

训练前唯一配置改动是将`configs/harl_mappo_short.yaml`中的`checkpoint.keep_last`从3调整为5，以满足本次必须同时保留update 1、3、5的验收要求。该改动不影响采样、优化器、网络、环境或学习语义。

冻结Suite：

```text
C:\Users\bulio\Desktop\IDC\ultimate_simplify\evaluation_suites\smoke_v1
SHA-256 = b74632bcb33a8fc00f682668e1677307a03376b0c2188f2afc29fdcb4d80593a
```

Grid CSV存在，训练metadata与Suite均记录：

```text
1c3319eadc5d93e9f339eefbe3824e0e655c1ad2bedb1c43d2ba08a16dc2296e
```

项目HEAD：`e3e77c487a1e59ed6d891e5ae34fe4732c580754`，dirty=true。HARL HEAD：`050ad6a294fe9f7572985dea910d59ea6d4f94b4`，dirty=false。两者均写入训练metadata和checkpoint provenance。

### A.2 正式训练命令

```powershell
$env:HARL_SOURCE_PATH="C:\Users\bulio\Desktop\IDC\HARL"
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'

& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" `
  train/train_harl_mappo_short.py `
  --config configs/harl_mappo_short.yaml `
  --harl-source "C:\Users\bulio\Desktop\IDC\HARL" `
  --runtime-path ".tmp_harl_runtime" `
  --seed 7110 `
  --updates 5 `
  --episode-length 24 `
  --rollout-threads 2 `
  --checkpoint-interval 1 `
  --output-dir runs/part12_mappo_5update_gate `
  --scenario idc_bess_padding `
  --device cpu
```

训练退出码0。工具记录总wall time为202.5秒；runner metadata从`2026-07-28T14:46:21.815278Z`运行至`2026-07-28T14:49:21.275144Z`，内部训练阶段约179.46秒。

## B. 训练产物和计数

Run目录：

```text
C:\Users\bulio\Desktop\IDC\ultimate_simplify\runs\part12_mappo_5update_gate\gym\idc_bess_padding\mappo\idc_bess_mappo_short\seed-07110-2026-07-28-22-46-17
```

| 产物/计数 | 实际值 | 预期 | 结果 |
| --- | ---: | ---: | --- |
| `step_metrics.csv` | 240行 | 240 | 通过 |
| `episode_metrics.csv` | 10行 | 10 | 通过 |
| `update_metrics.csv` | 5行 | 5 | 通过 |
| global_step | 240 | 240 | 通过 |
| episodes_completed | 10 | 10 | 通过 |
| total_updates_completed | 5 | 5 | 通过 |
| status | completed | completed | 通过 |
| final actor files | 2 | 2 | 通过 |
| final critic files | 1 | 1 | 通过 |
| 中间checkpoint | update 1～5 | update 1～5 | 通过 |
| final checkpoint | `final.pt` | 存在 | 通过 |

每个update的两个worker均各有24步，hour严格为0～23，terminal各恰好一次。每个worker的episode_id独立按0～4递增；global step每update增加48，episode count每update增加2。子进程正常退出，最终Python进程数为0。

Reward一致性结果：

```text
max(abs(sum(24 step reward) - episode_reward)) = 0
max(abs(mean(two episode rewards) - rollout_episode_reward_mean)) = 0
max reward reconstruction error = 0
shared_reward_max_diff = 0
```

## C. 5-Update训练指标

| update | global step | episodes | episode reward mean | rollout s | update s | IDC clip frac | IDC KL | BESS clip frac | BESS KL | value loss | explained variance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48 | 2 | -21.454378 | 45.9876 | 0.0196 | 0.354167 | 0.017923 | 0 | 0.000439 | 53.0729 | -0.018590 |
| 2 | 96 | 4 | -23.139890 | 46.7931 | 0.0223 | 0.062500 | 0.007395 | 0 | 0.000517 | 53.9746 | -0.002518 |
| 3 | 144 | 6 | -18.955888 | 28.7399 | 0.0087 | 0.020833 | 0.004249 | 0 | 0.000160 | 37.5641 | 0.003328 |
| 4 | 192 | 8 | -25.401376 | 25.5865 | 0.0114 | 0 | 0.002463 | 0 | 0.000104 | 62.4563 | 0.005783 |
| 5 | 240 | 10 | -22.060878 | 25.0767 | 0.0087 | 0 | 0.002334 | 0 | 0.000077 | 49.5701 | 0.004620 |

5 updates没有单调性能趋势，也不应要求单调改善。Reward、动作、任务和物理指标均不是常数。

## D. Actor、Critic与Optimizer变化

Update 1 checkpoint中的Adam状态step=1且一、二阶矩存在。根据保存的Adam状态重建initial→update 1的实际参数步长：

| 模型 | 参数L2变化 | 最大绝对变化 | 说明 |
| --- | ---: | ---: | --- |
| IDC Actor | 0.0452234 | 0.000499998 | 非零、finite |
| BESS Actor | 0.0363037 | 0.000499996 | 非零、finite |
| Critic | 0.0482714 | 0.000499999 | 非零、finite |

BESS initial→update 1细分：physical mean row0 L2=`0.00287219`，physical log_std变化=`0.000499802`，shared trunk L2=`0.0361865`；virtual mean rows 1:22与virtual log_std 1:22变化均为0。

| 对比 | IDC Actor L2 | BESS Actor L2 | BESS physical mean L2 | BESS virtual mean L2 | BESS physical log_std | BESS virtual log_std | Critic L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| update 1→3 | 0.0543860 | 0.0478699 | 0.00389545 | 0 | 0.000960708 | 0 | 0.0891024 |
| update 3→5 | 0.0410225 | 0.0273836 | 0.00193000 | 0 | 0.000934243 | 0 | 0.0854894 |
| update 1→5 | 0.0883527 | 0.0738192 | 0.00580930 | 0 | 0.00189495 | 0 | 0.173533 |

固定输入下update 1→5输出最大变化：IDC Actor=`0.00433192`，BESS Actor=`0.00322217`，Critic value=`0.0240463`。三个optimizer在checkpoint 1、3、5中的Adam step分别严格为1、3、5，`exp_avg`与`exp_avg_sq`存在且finite。

结论：两个Actor和Critic均真实更新；BESS仅有效输出头与shared trunk更新，虚拟输出头参数保持初值。

## E. BESS有效动作Mask

| update | physical mean grad | physical log_std grad | shared trunk grad | virtual mean grad | virtual log_std grad | virtual entropy contribution | ratio diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.348398 | 0.025223 | 0.116883 | 0 | 0 | 0 | 0 |
| 2 | 2.522261 | 0.053937 | 0.086931 | 0 | 0 | 0 | 0 |
| 3 | 1.512350 | 0.072155 | 0.055498 | 0 | 0 | 0 | 0 |
| 4 | 0.645726 | 0.116562 | 0.025223 | 0 | 0 | 0 | 0 |
| 5 | 0.495396 | 0.043551 | 0.041597 | 0 | 0 | 0 | 0 |

每个update均满足：

```text
bess_mask_enabled = true
effective/padded/virtual dim = 1/22/21
effective ratio = physical dim0 ratio
virtual gradient = 0
virtual entropy optimization contribution = 0
IDC effective dim = 22
```

没有发现多worker广播错误。

## F. Reward、任务、BESS和Grid趋势

以下为每个update两个worker episode的均值，只描述工程趋势：

| upd | reward | completion | backlog | cost | carbon kg | peak kW | final SOC | charge kWh | discharge kWh | throughput kWh | max invalid kW | OPF/MEF | safe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | -21.4544 | 0.38697 | 616560.6 | 12493.8 | 12759.8 | 2091.9 | 0.28091 | 4560.3 | 6197.0 | 10757.3 | 364.6 | 1/1 | 0 |
| 2 | -23.1399 | 0.44514 | 476616.4 | 15207.9 | 14740.5 | 2106.6 | 0.59538 | 6437.2 | 4903.5 | 11340.7 | 280.2 | 1/1 | 0 |
| 3 | -18.9559 | 0.39109 | 566241.9 | 14644.3 | 14412.3 | 2057.1 | 0.61446 | 7186.7 | 5398.7 | 12585.4 | 328.2 | 1/1 | 0 |
| 4 | -25.4014 | 0.37528 | 620527.7 | 16245.2 | 15660.0 | 2323.6 | 0.70301 | 7419.9 | 4767.9 | 12187.8 | 47.3 | 1/1 | 0 |
| 5 | -22.0609 | 0.43934 | 514020.5 | 15805.7 | 15888.1 | 2160.0 | 0.70862 | 8300.9 | 5509.7 | 13810.6 | 192.4 | 1/1 | 0 |

训练240步中有127个充电步、112个放电步、1个空闲步。SOC下边界0.1出现3步，上边界0.9出现2步；BESS不可执行请求出现19步。它们没有随update持续单调增加，但应在40-update继续监控。

OPF与MEF全程成功率1.0；Grid penalty、Safe violation均为0。OPF/MEF cache hit/miss字段全程存在且连续增长，没有静默失败。

## G. PPO与Critic数值稳定性

所有训练CSV、模型参数和optimizer状态均finite。动作边界检查通过；训练BESS padded dim0范围`[0.126717, 0.865610]`，对应物理动作`[-0.746566, 0.731221]`。

IDC ratio从update 1的`[0.7240, 1.5382]`收敛至update 5的`[0.8549, 1.1615]`；clip fraction从0.3542降至0。BESS ratio在五个update中保持接近1，clip fraction始终0，KL从`4.39e-4`降至`7.68e-5`。没有持续极端ratio或KL升高。

Critic value loss范围37.56～62.46，finite且不呈持续快速上升；explained variance从-0.0186改善到小幅正值。需要监控的是裁剪前Critic grad norm每次均为116.44～133.60，明显高于`max_grad_norm=10`。IDC Actor裁剪前grad norm在update 1、2、5高于10；BESS Actor grad norm从3.35下降至0.499。

因此数值链稳定，但梯度裁剪趋势是`pass_with_monitoring`的主要原因。

## H. Checkpoint完整性

| 文件 | update | global step | episodes | size bytes | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `update_000001.pt` | 1 | 48 | 2 | 671045 | `54af183cfa7f6beb8378811b9e0a68cffc0ce263a7b1c66051378ad3c812e6f7` |
| `update_000002.pt` | 2 | 96 | 4 | 759749 | `16349c2b6f57b9ee20b8cc21722ba72c88290a41cb24a5622806b79dec91c004` |
| `update_000003.pt` | 3 | 144 | 6 | 839557 | `ed5e7ae5533e04ea3811681162b10d6835df3fb8d5bc399fb142345374287596` |
| `update_000004.pt` | 4 | 192 | 8 | 915717 | `cd2990063c9ba0c93884d87807bdcbebc44da4deb2736870ebf9cd7fc98dfc59` |
| `update_000005.pt` | 5 | 240 | 10 | 997061 | `beb0bd55c1aa476bc028f9c60fa3c254df5c28f18156de9064f60cc9997faa5e` |
| `final.pt` | 5 | 240 | 10 | 997061 | `beb0bd55c1aa476bc028f9c60fa3c254df5c28f18156de9064f60cc9997faa5e` |

所有manifest的size与SHA验证通过，worker seeds均为`[7110,8110]`，config fingerprint均为`8bd4fb8e6c4f96d79c793571d9e437951208dad648a9ee6e3f159075de1462dd`，Mask与并行拓扑一致。

Update 3执行了完整load-only resume preflight：调用现有`TrainingCheckpointManager.load()`恢复Actor、Critic、三个optimizer、RNG、两个worker环境、buffer与logger；兼容性校验通过，runner状态为`completed_updates=3`和`resumed=true`，随后立即关闭，`policy_updates_executed=0`。原checkpoint未改变。

## I. 固定Smoke评估

三个checkpoint均使用同一个冻结Suite、CPU、deterministic=true、单评估worker：

| checkpoint | step rows | episode rows | failures | internal seconds | actor unchanged/no grad/eval mode |
| --- | ---: | ---: | ---: | ---: | --- |
| update 1 | 72 | 3 | 0 | 75.42 | 通过 |
| update 3 | 72 | 3 | 0 | 99.09 | 通过 |
| update 5 | 72 | 3 | 0 | 75.78 | 通过 |

| upd | seed | reward | completion | backlog | cost | carbon kg | peak kW | final SOC | throughput | invalid | OPF/MEF | safe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 2026 | -16.5193 | 0.44959 | 433763.3 | 14016.1 | 13667.5 | 1050.2 | 0.52559 | 298.5 | 0 | 1/1 | 0 |
| 1 | 2027 | -16.3191 | 0.42595 | 492240.6 | 13765.9 | 13444.5 | 1035.0 | 0.52491 | 291.4 | 0 | 1/1 | 0 |
| 1 | 2028 | -11.6801 | 0.56795 | 279221.0 | 13303.0 | 13031.5 | 1007.3 | 0.52356 | 277.2 | 0 | 1/1 | 0 |
| 3 | 2026 | -16.8472 | 0.44910 | 434146.6 | 14638.7 | 14224.6 | 1086.9 | 0.61249 | 1184.1 | 0 | 1/1 | 0 |
| 3 | 2027 | -16.6114 | 0.42551 | 492611.8 | 14397.9 | 14009.8 | 1072.4 | 0.61173 | 1176.1 | 0 | 1/1 | 0 |
| 3 | 2028 | -11.9754 | 0.56720 | 279702.2 | 13928.0 | 13591.2 | 1044.2 | 0.61012 | 1159.2 | 0 | 1/1 | 0 |
| 5 | 2026 | -17.0271 | 0.44853 | 434591.4 | 14945.4 | 14498.0 | 1104.8 | 0.65576 | 1639.6 | 0 | 1/1 | 0 |
| 5 | 2027 | -16.7879 | 0.42509 | 492976.9 | 14708.9 | 14287.2 | 1090.6 | 0.65500 | 1631.5 | 0 | 1/1 | 0 |
| 5 | 2028 | -12.1575 | 0.56652 | 280144.9 | 14236.7 | 13866.7 | 1062.2 | 0.65334 | 1614.1 | 0 | 1/1 | 0 |

没有要求update 5优于update 1。三个场景中update 5 reward均略低，但这只说明策略改变后的短期Smoke轨迹，不是性能结论。

## J. Checkpoint间配对比较

同场景逐步比较的最大变化：

| 对比 | IDC action mean | IDC action min | IDC action max | BESS物理动作 | Reward | Grid power | SOC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1→3 | 0.000789 | 0.010179 | 0.011872 | 0.020109 | 0.133819 | 39.7562 | 0.086897 |
| 3→5 | 0.000614 | 0.009730 | 0.011571 | 0.009926 | 0.092924 | 19.4277 | 0.043271 |
| 1→5 | 0.001403 | 0.019399 | 0.023293 | 0.030004 | 0.226743 | 59.0292 | 0.130169 |

回答附件要求：

1. Checkpoint之间两个Actor参数不同；
2. 相同场景的IDC动作摘要与BESS真实动作轨迹均变化；
3. SOC、Grid power、cost、carbon、throughput等物理轨迹变化；
4. 不存在三个checkpoint输出完全相同的异常；
5. 变化有限且所有动作合法；
6. 固定评估无IDC动作饱和，SOC范围`[0.49854,0.65576]`，未撞0.1/0.9；
7. 三个checkpoint的OPF/MEF均100%成功。

## K. 确定性重复验收

Update 5×`scenario_000_seed_02026`额外重复一次：24 step、1 episode、0失败，内部耗时26.12秒。排除evaluation ID与计时字段后，Actor动作、Reward、所有Reward分量、任务、BESS、Grid、cache和final metrics的最大差异为0；没有把重复结果混入正式三场景aggregate。

## L. 信用分配初步诊断

IDC有效参数和固定动作轨迹均变化，任务完成率、backlog和服务指标随checkpoint产生小幅响应。

BESS physical row0、physical log_std与shared trunk持续更新；virtual输出头不更新。确定性BESS动作不是恒定值，但趋势为：

| checkpoint | physical mean | std | range | charge fraction | discharge fraction |
| --- | ---: | ---: | --- | ---: | ---: |
| update 1 | -0.00545 | 0.00273 | `[-0.00706, 0.00692]` | 95.83% | 4.17% |
| update 3 | -0.02444 | 0.00383 | `[-0.02717, -0.00680]` | 100% | 0% |
| update 5 | -0.03392 | 0.00429 | `[-0.03706, -0.01415]` | 100% | 0% |

SOC、throughput、cost、carbon和peak均随该动作趋势变化，说明环境对BESS第0维有响应，不属于Mask失效或信用分配阻塞。但update 3/5在三个Smoke场景中均为小幅持续充电，对小时、价格、SOC和负载的相关性较弱，应在40 updates重点监控是否继续单一化。此处不修改Reward或增加辅助loss。

## M. 测试与回归

### M.1 回归命令

```powershell
& "C:\Users\bulio\miniconda3\envs\idc_ppo\python.exe" -m pytest `
  marl/tests/test_harl_mappo_formal_entry.py `
  marl/tests/test_bess_effective_action_mask.py `
  marl/tests/test_training_metrics_logger.py `
  marl/tests/test_training_checkpoint_resume.py `
  marl/tests/test_parallel_rollout_sampling.py `
  marl/tests/test_fixed_evaluation.py -q
```

绑定了真实Part9连续/恢复、Part10两/四worker和本次Part12固定评估产物。

```text
89 passed
0 failed
0 skipped
1137 warnings
pytest = 202.43 s
tool wall = 206.6 s
```

Warnings为1120条pandapower `tap_dependency_table`弃用提示，以及1条pytest cache权限提示在各测试上下文中的汇总；没有功能warning。

### M.2 训练与评估命令异常记录

1. 首个多worker preflight曾从stdin启动Python，和Windows `spawn`不兼容，120秒超时；残留PID 7320随后被明确终止。正式训练改用附件指定的真实脚本文件入口，不涉及代码修复。
2. 第一次update 1评估命令遗漏`--runtime-path .tmp_harl_runtime`，在模型加载和场景运行前即因本地运行依赖不可见退出1；补齐正式入口已有参数后成功。未生成失败评估产物，也未修改模型。
3. 唯一一次5-update正式训练退出0；没有重跑seed、没有第二次5-update、没有40-update。
4. 最终残留Python进程数为0。

## N. 问题分类

### 阻塞40-update

无。

### 40-update中必须监控

- Critic每个update均触发较大的梯度裁剪；持续观察value loss、explained variance和裁剪前grad norm。
- IDC Actor在update 1、2、5的裁剪前grad norm超过10；观察clip fraction、KL和ratio范围。
- BESS固定策略在update 3/5偏向小幅持续充电；观察充放电/空闲比例、价格与SOC响应。
- 训练中SOC触边5/240步、不可执行请求19/240步；观察是否随训练持续增加。
- 5-update固定评估Reward没有改善；只能在更多updates和正式Validation协议中判断。

### 正式实验前必须完成

- 物化并冻结Validation Suite；Test Suite仍需保持未使用。
- 预先确定checkpoint选择主指标，禁止用训练Reward临时挑模型。
- 冻结3-seed运行协议、命名与失败处理规则。

### 建议改进

- 正式评估命令模板固定带`--runtime-path .tmp_harl_runtime`或设置`HARL_RUNTIME_PATH`。
- Gate工具可后续正式化为load-only resume子命令，避免一次性预检脚本。

### 可以保持现状

- 双worker采样、seed策略、bounded Box、BESS Mask、Reward重建、日志schema、checkpoint语义、固定Suite与确定性评估。
- 环境物理、Reward、MAPPO loss、GAE、网络、学习率、clip与entropy参数均无需因本次结果修改。

## O. Gate Summary

结构化结果保存于：

```text
C:\Users\bulio\Desktop\IDC\ultimate_simplify\evaluations\part12_gate\gate_summary.json
```

核心字段：

```json
{
  "training_completed": true,
  "total_transitions": 240,
  "episodes_completed": 10,
  "checkpoint_integrity_passed": true,
  "logging_consistency_passed": true,
  "mask_invariants_passed": true,
  "deterministic_evaluation_passed": true,
  "deterministic_repeat_max_difference": 0.0,
  "opf_success_passed": true,
  "mef_success_passed": true,
  "finite_values_passed": true,
  "actors_updated": {
    "idc": true,
    "bess_effective": true,
    "bess_virtual_output_head_unchanged": true
  },
  "critic_updated": true,
  "policy_changed_across_checkpoints": true,
  "blocking_issues": [],
  "gate_decision": "pass_with_monitoring"
}
```

## P. 第十二部分判断

| 检查项 | 判断 |
| --- | --- |
| 训练链完整性 | 通过 |
| 样本和日志计数 | 通过，240/10/5精确 |
| Actor/Critic更新 | 通过，均有限且非零 |
| BESS Mask | 通过，五个update全部满足 |
| PPO数值稳定性 | 基本通过；Critic/IDC梯度裁剪需监控 |
| 环境与Grid稳定性 | 通过，OPF/MEF 100%，无Safe/Grid penalty |
| Checkpoint完整性 | 通过，1～5及final完整，update 3 load-only通过 |
| 固定评估确定性 | 通过，重复最大差异0 |
| Checkpoint间策略变化 | 通过，参数、动作和物理轨迹均变化 |
| 信用分配初步状态 | 基本正常；BESS charge-only趋势需监控 |

最终选择：

```text
2. 基本通过，可以进入40-update但需重点监控
gate_decision = pass_with_monitoring
```

不存在必须先修复的训练阻塞问题。

## Q. 进入40-update前最小建议

1. 继续使用同一正式入口、2 workers、seed协议、bounded Box、BESS Mask和当前超参数，不因本次短期Reward变化调参。
2. 40-update日志中持续监控Critic/IDC裁剪前梯度、value loss、KL、ratio和clip fraction。
3. 持续监控BESS充/放/空闲比例、SOC触边、不可执行请求，以及动作对小时、价格、SOC和负载的响应。
4. 保证checkpoint 0/10/20/30/40与final均保留，并继续用同一`smoke_v1`做工程评估。
5. 正式模型选择前再物化Validation Suite、冻结选择指标和3-seed协议；本部分不执行该步骤。

本次未运行40 updates、未物化Validation/Test、未调整Reward或超参数、未进入HAPPO、未实现HGTA或Safe RL。
