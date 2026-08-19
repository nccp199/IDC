# MAPPO短训练前检查——第七部分：BESS Padding与虚拟动作诊断

## 0. 范围、方法与最终结论

本次仅执行：

- Bridge、padding、HARL Actor、distribution、buffer、MAPPO/HAPPO loss 的只读源码追踪；
- 四组 fresh、同 seed、24步固定动作的物理不变性回归；
- 512次 BESS Actor 纯 forward 采样；
- 一次24步 rollout 和唯一一次正式 MAPPO update 的梯度/ratio诊断。

未修改 Bridge、padding、Actor、distribution、buffer、MAPPO/HAPPO loss、动作空间、环境、Reward、Grid/Safe或配置；未保存模型、未运行5-update或40-update训练。

诊断使用：

- 项目：`C:/Users/bulio/Desktop/IDC/ultimate_simplify`
- HARL：`C:/Users/bulio/Desktop/IDC/HARL`
- Python：`C:/Users/bulio/miniconda3/envs/idc_ppo/python.exe`
- seed：`7110`
- episode length：`24`
- PPO epoch / mini-batch：`1 / 1`
- `action_aggregation=prod`
- `entropy_coef=0.01`
- `max_grad_norm=10.0`

正式runner的依赖目录中 `yaml`、`tensorboardX` 和 `setproctitle` 只能被当前沙箱识别为空namespace。诊断最终采用现有 smoke 已验证的配置字典，并只在进程内将真实 `torch.utils.tensorboard.SummaryWriter` 暴露为 `tensorboardX.SummaryWriter`、将进程标题设置替换为 no-op；这两个shim不参与数值计算。前面的导入/初始化失败均未进入rollout或update，最终成功运行的update严格为1次。

### 总结判断

- **物理层：通过。** 虚拟21维对 transition、Reward、OPF、MEF、cache、SOC、功率和真实BESS动作的最大影响为 `0.0`。
- **Observation接口层：数值稳定，但不是完全惰性。** padding输入始终为0；当前 `LayerNorm(288)` 会把124个零padding位置变成同一个非零标准化值。
- **算法层：未通过。** 虚拟21维完整进入 log-prob、entropy、PPO ratio、clip、KL和梯度，并显著主导BESS更新。
- **第七部分综合选择：4——未通过，虚拟维显著干扰BESS学习。** 按本检查给出的阻塞标准，不建议在修正概率/loss语义前开始正式MAPPO短训练。

关键实测证据：

| 指标 | 22维联合BESS | 仅物理第0维诊断 | 差异 |
| --- | ---: | ---: | ---: |
| update后 ratio std | 0.22242 | 0.02897 | 7.68倍 |
| update后最大ratio偏差 | 0.36293 | 0.06666 | 5.45倍 |
| clip fraction | **37.5%** | **0%** | 37.5个百分点 |
| approximate KL | 0.024829 | 0.000423 | 58.7倍 |
| mean-head梯度范数 | 15.5006（虚拟联合） | 1.7884（物理） | 8.67倍 |
| shared trunk梯度范数 | 0.4922 | 0.0695 | 7.08倍 |
| 总Actor梯度范数 | 15.6144 | 1.7909 | 8.72倍 |
| entropy贡献 | 虚拟 `-8.4157` | 物理 `-0.5529` | 虚拟占联合绝对量约93.8% |

---

## A. Padding调用链

| 层级 | 文件/函数 | 输入shape | 输出shape | 虚拟维处理 |
| --- | --- | ---: | ---: | --- |
| 双Agent环境 | `marl/envs/idc_grid_multi_agent_env.py` `IDCGridMultiAgentEnv.reset()/step()` | Grid obs `(288,)`、IDC `(22,)`、BESS `(1,)` | IDC obs `(288,)`、BESS obs `(164,)`、state `(294,)` | 尚无虚拟维 |
| 原始Bridge | `marl/bridges/harl_bridge.py` `HarlIDCGridBridge.reset()/step()` | 异构obs/action | obs `((288,), (164,))`，action `((22,), (1,))` | 保持异构 |
| Padding Bridge | `marl/bridges/harl_padded_bridge.py` `_pad_observations()` | BESS obs `(164,)` | BESS obs `(288,)` | 尾部补124个0 |
| Padded spaces | 同文件 `_homogenize_box_spaces()` | action spaces `(22,)`、`(1,)` | 两个Box均为 `(22,)`、bounds `[0,1]` | BESS新增21个合法Box维 |
| VecEnv reset | HARL `harl/envs/env_wrappers.py` `ShareDummyVecEnv.reset()` | 单环境 `(2,288)` | `(1,2,288)` | 完整stack |
| Actor trunk | HARL `harl/models/policy_models/stochastic_policy.py` `StochasticPolicy.forward()` | `(1,288)` | hidden `(1,32)` | BESS的288维整体进入MLP |
| ACTLayer | HARL `harl/models/base/act.py` `ACTLayer.forward()` | hidden `(1,32)` | action/log-prob `(1,22)` | 22维全部生成和采样 |
| bounded distribution | HARL `harl/models/base/distributions.py` `BoundedDiagGaussian.forward()` | mean/std `(1,22)` | `AffineTanhNormal` | 22维均为独立Gaussian分量 |
| Runner collect | HARL `harl/runners/on_policy_base_runner.py` `collect()` | 两个actor各 `(1,22)` | actions/log-probs `(1,2,22)` | `np.array()`要求同形 |
| Actor buffer | HARL `harl/common/buffers/on_policy_actor_buffer.py` | 每步BESS `(1,22)` | episode `(24,1,22)` | 保存完整动作和逐维log-prob |
| evaluate | `StochasticPolicy.evaluate_actions()` → `ACTLayer.evaluate_actions()` | batch action `(24,22)` | new log-prob `(24,22)`、entropy scalar | 22维全部评估 |
| MAPPO ratio | HARL `harl/algorithms/actors/mappo.py` `MAPPO.update()` | old/new `(24,22)` | ratio `(24,1)` | 22个逐维ratio相乘 |
| HAPPO ratio | HARL `harl/algorithms/actors/happo.py` `HAPPO.update()` | old/new `(batch,22)` | ratio `(batch,1)` | 同样相乘，再乘HAPPO factor |
| 动作反向还原 | `HarlPaddedBridge._dehomogenize_actions()` | `(2,22)` | IDC `(22,)`、BESS effective `(1,)` | BESS `[1:22]`丢弃 |
| 真实动作拼接 | `marl/adapters/action_adapter.py` `FlatActionAdapter.compose_action()` | IDC `(22,)`、BESS `(1,)` | legacy flat `(23,)` | BESS第0维位于flat最后一维 |
| 底层物理映射 | `envs/idc_price_env.py` `IDCPriceEnv20D.step()` | BESS scalar `a∈[0,1]` | `2a-1∈[-1,1]` | `<0.5`充电，`0.5`空闲，`>0.5`放电 |

Agent顺序在 `marl/specs/agent_specs.py` 固定为：

```text
agent 0 = idc
agent 1 = bess
```

`HarlIDCGridBridge.agent_order`、`HarlPaddedBridge.agent_order`、runner actor/buffer索引和固定测试均使用同一顺序，未发现动作错位。

---

## B. Observation Padding

### B.1 BESS真实164维组成

构造位置：`marl/observations/bess_obs_builder.py` `BESSObservationBuilder.build()`。

| BESS区间 | 原始来源 | 物理含义 | 维数 |
| --- | --- | --- | ---: |
| `[0:6]` | wrapped obs `[0:6]` | 当前温度、电价、到达量、队列、time sin、time cos | 6 |
| `[6:30]` | base obs `[136:160]` | 24小时电价预测 | 24 |
| `[30:54]` | base obs `[160:184]` | 24小时环境温度预测 | 24 |
| `[54:78]` | base obs `[184:208]` | 24小时任务到达量预测 | 24 |
| `[78:102]` | base obs `[208:232]` | 24小时PV预测 | 24 |
| `[102:126]` | base obs `[232:256]` | 24小时time sin | 24 |
| `[126:150]` | base obs `[256:280]` | 24小时time cos | 24 |
| `[150:158]` | Grid obs `[280:288]` | LMP、MEF+、MEF-、最小电压、最大线路负载、网损、安全惩罚、OPF成功 | 8 |
| `[158:164]` | supplemental | SOC、BESS能量、IDC功率、购电功率、充电功率、放电功率 | 6 |
| **合计** |  |  | **164** |

基础观测维数来自 `envs/idc_price_env.py`：当前特征 `6+10+6×20=136`，forecast `6×24=144`，合计280；`GridCoupledEnv` 再追加8维成为288。

### B.2 Padding实现

`HarlPaddedBridge._pad_observations()` 明确执行：

```python
padded_bess = np.pad(bess, (0, 124), constant_values=0)
```

因此：

```text
padded_obs[1, :164]   = BESS真实观测
padded_obs[1, 164:288] = 124个float32零
```

检查结果：

- reset、24个step和VecEnv自动reset后的buffer尾部最大绝对值均为 `0.0`；
- dtype始终为 `float32`；
- 两个相同真实观测padding后逐值相等；
- 只修改真实索引7，padding区最大变化为 `0.0`；
- `np.pad`创建新数组，修改返回padding区不会反向修改164维源数组；
- terminal observation先由同一函数padding，再被VecEnv复制为 `info[0]["original_obs"]`；自动reset返回的新观测也再次走同一padding函数。

### B.3 Feature normalization

正式配置：

| 配置/机制 | 当前值 | 是否改变输入padding |
| --- | --- | --- |
| `use_feature_normalization` | `true` | 是，启用 `nn.LayerNorm(288)` |
| `use_obs_norm` | 不存在/未启用 | 否 |
| running mean/std | 不存在 | 否 |
| MLP hidden LayerNorm | 每层32维hidden后存在 | 不直接改输入向量 |
| `use_valuenorm` | `false` | 与observation无关 |

`MLPBase.forward()` 在第一层线性层前执行 `feature_norm=LayerNorm(288)`。LayerNorm按每个样本的288维共同均值和方差进行仿射变换：

\[
y_i=\gamma_i\frac{x_i-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}+\beta_i.
\]

所以即使 `x[164:288]=0`，只要真实164维的均值不为0，标准化后的padding位置就不会是0。

fresh BESS Actor实测：

| 指标 | 数值 |
| --- | ---: |
| padding输入最大绝对值 | 0.0 |
| LayerNorm后padding均值 | -0.05955596 |
| LayerNorm后paddingmin/max | -0.05955595 / -0.05955595 |
| 只改一个真实字段后padding标准化值最大变化 | 1.48e-6 |
| NaN/inf | 无 |

这属于“输入padding维被LayerNorm改变”，不同于“线性层bias使hidden非零”。124个位置在LayerNorm后都变成相同但非零的值，并拥有独立可训练的 `gamma/beta`。一次全22维loss诊断中，padding区LayerNorm参数梯度范数为 `0.02523`，真实164维区为 `0.03326`；padding区并非完全惰性。

该现象数值稳定，不构成物理错误，但未来异构/掩码设计应一并考虑“只对真实164维归一化后再padding”或等价feature mask。

---

## C. Action Padding

### C.1 正式动作空间

| Agent | HARL action shape | bounds | 真实物理维数 | 虚拟维数 |
| --- | ---: | --- | ---: | ---: |
| IDC | `(22,)` | 每维 `[0,1]` | 22 | 0 |
| BESS | `(22,)` | 每维 `[0,1]` | 1 | 21 |

`_homogenize_box_spaces()`将原始IDC `(22,)` 和BESS `(1,)` Box合并为同一个 `(22,)` Box。BESS的全部22维因此都被Actor和distribution当作真实有效的有界连续维。

### C.2 动作还原

实际逻辑：

```text
BESS padded action shape (22,)
→ bess_full_action = action[1].copy()
→ bess_effective_action = bess_full_action[:1].copy()
→ FlatActionAdapter验证shape=(1,), finite且位于[0,1]
→ 拼到legacy flat action最后一维
→ 底层以2a-1映射充/空闲/放电
```

结论：

- 只有BESS第0维进入物理环境；
- 第1～21维直接丢弃；
- 不存在平均、求和、最大值选择、reshape混合或由虚拟维修改第0维；
- padding层不clip任何动作；
- 任意维含NaN/inf时，`HarlPaddedBridge`立即拒绝；
- 正式runner在环境step前检查全部22维是否满足Box bounds；
- 直接绕过runner调用Bridge时，虚拟维只检查finite，不单独检查bounds，因为它们随后被丢弃；有效第0维仍由`FlatActionAdapter`严格拒绝越界；
- 正式Actor使用Affine-Tanh bounded distribution，512×22次采样全部位于 `[0,1]`。

---

## D. 物理不变性测试

### D.1 设计

使用正式 `make_harl_single_env()` 创建4个fresh、seed均为7110的环境。每步保持：

- IDC 22维动作全部为0.5；
- BESS第0维全部为0.5；
- 只改变BESS `[1:22]`：全0、全1、固定随机、每步随机。

每个环境运行完整24步。比较reset以及每步返回的完整 observation、state、reward、done、全部 `info` 数值和非数值字段。

### D.2 结果

| 项目 | 结果 |
| --- | ---: |
| reset最大数值差 | 0.0 |
| 24步transition最大数值差 | **0.0** |
| 非数值字段不一致数量 | 0 |
| padding尾部最大绝对值 | 0.0 |
| 底层真实BESS动作 | 四组全部精确为0.5 |
| terminal done | 四组全部在第24步为true |
| Reward分量数量 | 23，全部逐值一致 |
| NaN/inf | 无 |

因此以下字段全部最大差异为0：

- reward及全部Reward components；
- observation、centralized state、done；
- 完成工作、任务完成、backlog、deadline、SLA；
- IDC功率、BESS charge/discharge、SOC；
- cost、carbon、`P_bus_net`、购电功率；
- LMP、MEF、电压、线路负载、OPF/MEF状态；
- cache enabled/hit/count/rate/size/bin诊断。

`info`不直接公开内部hash key；源码 `grid_model/grid_cache.py` 的 `make_opf_key()` / `make_mef_key()`只接收物理网格状态与负载等参数，虚拟动作从未到达该层。四个环境最终OPF/MEF miss count均为25、cache size均为25，所有cache诊断逐值一致。

物理层验收通过：

```text
只改变BESS虚拟21维
→ 所有可观测物理、Reward和cache结果最大差异 = 0
```

---

## E. Buffer与概率计算

### E.1 Buffer

唯一一次rollout后，BESS `OnPolicyActorBuffer` 实际形状为：

| 字段 | shape/值 | 含义 |
| --- | --- | --- |
| `obs` | `(25,1,288)` | 24步加末状态，padding尾部仍全0 |
| `actions` | `(24,1,22)` | 保存第0物理维和21虚拟维 |
| `action_log_probs` | `(24,1,22)` | 保存逐维old log-prob，尚未求和 |
| `available_actions` | `None` | 连续Box无离散availability mask |
| `active_masks` | `(25,1,1)` | agent存活mask，不是动作维mask |

当前没有任何 virtual-action mask。

### E.2 Bounded Gaussian

`BoundedDiagGaussian`为每一维产生：

- 独立mean `μ_j(o)`：`fc_mean`的第j行；
- 独立可学习 `log_std[j]`；
- 相同std参数化公式，但不是共享的单一std参数；
- latent `z_j ~ Normal(μ_j,σ_j)`；
- action `a_j = center_j + half_range_j·tanh(z_j)`。

对 `[0,1]` Box，`center=0.5`、`half_range=0.5`。含Jacobian修正的逐维log-prob为：

\[
\log\pi_j(a_j|o)=
\log\mathcal{N}(z_j;\mu_j,\sigma_j)
-\log(half\_range_j)
-\log(1-\tanh^2 z_j).
\]

`ACTLayer.forward()`对Box不求和，返回 `(batch,22)`。数学上的BESS联合概率为：

\[
\log\pi_{BESS}(a|o)=\sum_{j=0}^{21}\log\pi_j(a_j|o).
\]

deterministic action同样输出22个mode，虚拟维仍存在。

### E.3 PPO ratio

正式配置 `action_aggregation=prod`。`MAPPO.update()`实际执行：

\[
r_j=\exp(\log\pi^{new}_j-\log\pi^{old}_j),
\qquad
r_{full}=\prod_{j=0}^{21}r_j
=\exp\left(\sum_{j=0}^{21}\Delta\log\pi_j\right).
\]

随后只对这个联合ratio做 `[0.8,1.2]` clipping。因此即使虚拟动作完全不影响环境，虚拟策略变化仍会改变整个BESS ratio、clip选择和policy loss。

HAPPO使用同样的逐维ratio聚合，之后再乘顺序更新factor；虚拟维问题会原样继承到HAPPO。

### E.4 Entropy

Affine-Tanh没有在此实现解析熵；`entropy(action)`使用当前动作的样本估计：

\[
H_{sample}=-\sum_{j=0}^{21}\log\pi_j(a_j|o).
\]

所以BESS entropy统计包含1个物理维和21个虚拟维，当前无法从HARL默认train info中拆分。

本次rollout的update前结果：

| 项目 | 值 |
| --- | ---: |
| 物理第0维entropy | -0.55290 |
| 虚拟21维entropy sum | -8.41573 |
| 完整22维entropy | -8.96863 |
| 虚拟维占联合绝对量 | 约93.8% |

连续分布的微分熵允许为负。Affine-Tanh将概率密度压到有限区间，密度可大于1，因而 `-log density` 的样本均值可以为负；这里的 `-8.97` 本身不是实现错误。

---

## F. 虚拟维梯度诊断

### F.1 输出头定位

BESS Actor参数位置：

| 参数 | shape | 物理/虚拟切分 |
| --- | ---: | --- |
| `act.action_out.fc_mean.weight` | `(22,32)` | row0物理；row1:22虚拟 |
| `act.action_out.fc_mean.bias` | `(22,)` | index0物理；index1:22虚拟 |
| `act.action_out.log_std` | `(22,)` | index0物理；index1:22虚拟 |
| `base.*` | shared trunk | 所有22维梯度共同回传 |

### F.2 update前分项autograd

在正式loss不变的前提下，对同一rollout batch只读地分别计算联合loss和仅物理第0维的counterfactual loss。该诊断不调用optimizer、不写回参数。

| 梯度范数 | 正式22维总loss | 仅物理维总loss | 联合/物理 |
| --- | ---: | ---: | ---: |
| mean物理row0 | 1.78838 | 1.78839 | 1.00 |
| mean虚拟21维联合 | **15.50061** | 0 | — |
| mean虚拟每维平均 | 2.77576 | 0 | — |
| log-std物理维 | 0.06553 | 0.06553 | 1.00 |
| log-std虚拟联合 | **0.31074** | 0 | — |
| shared trunk | **0.49223** | 0.06954 | 7.08 |
| 总Actor | **15.61443** | 1.79094 | 8.72 |

虚拟mean-head联合梯度是物理mean row的8.67倍；即使按每个虚拟维平均，仍约为物理row的1.55倍。

### F.3 梯度来源

| loss来源 | 物理mean | 虚拟mean联合 | trunk | 总梯度 |
| --- | ---: | ---: | ---: | ---: |
| policy gradient | 1.78485 | **15.51287** | 0.49378 | 15.62625 |
| `-0.01×entropy` | 0.01382 | 0.14257 | 0.00429 | 0.14335 |
| 两者合计 | 1.78838 | **15.50061** | 0.49223 | 15.61443 |

本次主要干扰来源是联合PPO ratio中的policy gradient，而不是entropy bonus；entropy仍额外给全部虚拟维梯度。

虚拟输出头梯度通过共享BESS Actor trunk回传，因而会修改真实第0维未来依赖的hidden representation。虽然当次输出row0的直接梯度不被虚拟row数值相加，但共享trunk和后续optimizer state均受到虚拟目标影响，能够间接改变物理策略。

### F.4 唯一一次正式update

| 指标 | IDC Actor | BESS Actor |
| --- | ---: | ---: |
| policy loss | -4.97e-9 | 7.45e-8 |
| entropy | -8.97085 | -8.96863 |
| pre-clip actor grad norm | 13.60961 | **15.61443** |
| max grad norm | 10.0 | 10.0 |
| reported ratio（update执行时） | 1.0 | 0.99999994 |

两Actor的pre-clip梯度均超过10并被裁剪。hook保存的tensor随后被in-place gradient clipping缩放，因此hook切片对应post-clip梯度；未裁剪的分维数值以上述update前同一正式loss autograd结果为准。

critic本次 `value_loss=82.4373`、`critic_grad_norm=139.996`；均为有限值。本部分不据此修改critic或ValueNorm。

### F.5 update后联合与物理ratio

对完成一次optimizer step后的同一rollout动作重新evaluate：

| 指标 | 联合22维ratio | 仅物理第0维ratio |
| --- | ---: | ---: |
| mean | 1.03061 | 0.99851 |
| std | **0.22242** | 0.02897 |
| min | 0.66524 | 0.93334 |
| max | 1.36293 | 1.06249 |
| 最大绝对偏差 | **0.36293** | 0.06666 |
| clip fraction | **0.375** | **0.0** |
| approximate KL | **0.024829** | 0.000423 |

其他诊断：

- 联合ratio与物理ratio平均绝对gap：`0.18202`；
- clip disagreement：`37.5%`；
- full surrogate diagnostic loss：`-0.09425`；
- physical-only surrogate diagnostic loss：`-0.00476`；
- 二者差：`-0.08950`。

结论不是“虚拟维只有轻微统计噪声”，而是它们实际决定了相当一部分clip和surrogate更新。

---

## G. IDC/BESS统计对比

| 指标 | IDC Actor | BESS Actor |
| --- | ---: | ---: |
| HARL动作维数 | 22 | 22 |
| 物理动作维数 | 22 | 1 |
| 虚拟维数 | 0 | 21 |
| buffer log-prob维数 | 22 | 22 |
| entropy聚合维数 | 22 | 22 |
| Actor总参数 | 11,756 | 11,756 |
| base/trunk参数 | 11,008 | 11,008 |
| mean-head参数 | 726 | 726 |
| log-std参数 | 22 | 22 |
| BESS物理输出参数 | — | 34 |
| BESS虚拟输出参数 | — | 714 |
| 1-update entropy | -8.97085 | -8.96863 |
| 1-update pre-clip gradient norm | 13.60961 | 15.61443 |

BESS Actor为了1个物理维保留了714个虚拟输出参数，真实物理输出只占34个参数。两个Actor的总entropy表面上相近，但BESS的约93.8%来自无物理作用的虚拟维。

同一个 `entropy_coef=0.01` 对BESS不是“单物理维探索”，而是22个分量的entropy和。512次fresh forward中：

| 分布指标 | 物理第0维 | 虚拟21维整体 |
| --- | ---: | ---: |
| mean | 0.51013 | 0.50047 |
| std | 0.16039 | 0.16117 |
| min | 0.10223 | 0.04936 |
| max | 0.89862 | 0.93038 |

21个虚拟维各自mean约 `0.486～0.515`，std约 `0.154～0.169`；无样本饱和到 `1e-6` 或 `1-1e-6`，全部在bounds内。虚拟维平均绝对相关系数为 `0.0360`、最大为 `0.1213`，与独立采样的有限样本波动一致。

虚拟维会消耗PyTorch RNG。相同seed下，实测22维采样和只构造1维采样得到的首个物理样本也不同（`0.60703` vs `0.56640`），后续随机数同样不同。这意味着未来改变动作头维数会改变随机序列，不能期待与padding-v1逐步复现一致。

---

## H. Checkpoint结构影响

BESS checkpoint保存完整 `StochasticPolicy.state_dict()`，其中固定包含：

| state key | 当前shape |
| --- | ---: |
| `act.action_out.fc_mean.weight` | `(22,32)` |
| `act.action_out.fc_mean.bias` | `(22,)` |
| `act.action_out.log_std` | `(22,)` |
| `act.action_out.action_low` | `(22,)` |
| `act.action_out.action_high` | `(22,)` |

如果未来改成1维BESS动作头，以上键都会尺寸不匹配，旧checkpoint不能直接strict load。即使使用 `strict=False`，PyTorch仍会对同名但shape不同的tensor报错；迁移时需要先过滤这些键，再显式复制mean row0、bias0、log-std0和bounds，trunk参数可继续加载。

当前正式metadata记录：

- `agent_order=[idc,bess]`；
- `observation_dims=[288,288]`；
- `action_dims=[22,22]`；
- `state_dim=294`；
- bounded Box开关及aggregation。

但未记录：

- BESS真实action dim=1；
- padded action dim=22；
- virtual dims=21；
- observation真实dim=164；
- padding策略版本。

因此当前短训练checkpoint应被明确视为 **`padding-v1 / bess-output-22`** 结构。唯一一次诊断没有调用 `runner.save()`，`runs/part7_diag` 中没有 `.pt` checkpoint。

---

## I. 解决方案评估

### 方案A：保持当前22维联合PPO

优点：改动最少，与现有接口和checkpoint兼容。

实测缺点：虚拟维占entropy约93.8%，虚拟mean梯度联合为物理维8.67倍，联合KL约为物理维59倍，并造成37.5% clip disagreement。

**判断：不接受作为正式MAPPO短训练方案。**

### 方案B：虚拟维loss mask

目标：环境和buffer仍保持22维，但BESS mask为 `[1,0,...,0]`，IDC mask全1。数学上应执行：

\[
r=\exp\left(\sum_j m_j(\log\pi^{new}_j-\log\pi^{old}_j)\right),
\qquad
H=-\sum_j m_j\log\pi_j.
\]

当前distribution和buffer已经保留逐维log-prob，因此实现基础较好；需要把agent-specific action mask正式传入 `ACTLayer.evaluate_actions()`、MAPPO/HAPPO aggregation和entropy，并确保诊断、metadata与测试覆盖。可能涉及：

- HARL `harl/models/base/act.py`；
- HARL `harl/algorithms/actors/mappo.py`；
- HARL `harl/algorithms/actors/happo.py`；
- HARL `harl/algorithms/actors/on_policy_base.py`；
- 项目 `marl/runners/idc_mappo_runner.py`；
- 配置、metadata和新回归测试。

buffer可继续保存完整22维old log-prob，不必先改schema。采样/RNG和checkpoint仍保持22维，但虚拟输出头将不再获得policy/entropy梯度。

**判断：最小且数学一致的短期正式方案。**

### 方案C：BESS只学习1维，padding层补常数

语义最清晰，但现有stock runner的 `collect()` 使用：

```python
np.array(action_collector).transpose(1, 0, 2)
```

IDC `(1,22)` 与BESS `(1,1)`不能直接组成规则数组。需要在collect后、进入VecEnv前补齐动作，同时对buffer和evaluate保留真实1维；这已是runner适配，不只是Bridge改动。旧22维checkpoint也需迁移。

**判断：长期比loss mask更清晰，但不是最小改动。**

### 方案D：正式异构动作空间

HARL的非共享Actor构造和每Agent `OnPolicyActorBuffer`本身可以接收各自action space；真正阻塞位于stock runner action收集的 `np.array()`、VecEnv stack以及统一数组接口。项目当前padding正是为了绕开这些同形要求。

完全异构支持需要自定义runner collect/insert/eval路径或上游HARL扩展，并重新定义checkpoint兼容。

**判断：架构上最正确，适合HAPPO/HGTA前统一规划，不适合当前最小修正。**

### 方案E：保持现状先跑短训练

原本可接受条件包括物理不变、数值有限、虚拟梯度不严重、ratio/clip不失控。前两项通过，但后两项未通过：虚拟维显著主导梯度、KL和clip。

**判断：拒绝。**

---

## J. 问题分类

### J.1 阻塞MAPPO短训练

1. BESS虚拟21维进入联合ratio并使37.5%样本发生“联合clip而物理维不clip”。
2. 虚拟mean-head联合梯度是物理mean的8.67倍，总联合梯度是physical-only诊断的8.72倍。
3. 虚拟目标通过共享trunk产生约7.08倍于physical-only的trunk梯度，能够间接改变真实第0维策略。
4. 联合approximate KL是物理维约58.7倍，已经不是轻微统计影响。

### J.2 必须在正式长训练/HAPPO前修正

1. 为MAPPO和HAPPO统一定义agent-specific有效动作mask或异构动作语义。
2. entropy、ratio、KL、clip和policy loss使用同一有效维集合。
3. checkpoint metadata记录真实/padded/virtual dims和padding版本。
4. 日志分别记录物理维与虚拟维ratio、entropy、gradient和clip。
5. 处理BESS observation LayerNorm对零padding的仿射影响，至少明确其设计和版本。

### J.3 建议改进

1. 新增BESS physical-only ratio与联合ratio的持续诊断；
2. 新增每维mean/log-std梯度诊断；
3. 明确padding-v1 checkpoint迁移脚本/规则；
4. 在最终异构实现中减少无效RNG消费和714个虚拟输出参数。

### J.4 可以保持现状

1. `HarlPaddedBridge`物理动作切片逻辑；
2. BESS真实第0维在底层的充/空闲/放电映射；
3. centralized state 294维不随local observation padding拼接；
4. bounded affine-tanh及Jacobian修正本身；
5. 物理不变性和cache key的现有实现；
6. buffer保留逐维log-prob这一数据能力。

---

## K. 第七部分判断

### 物理层判断

**通过。** 虚拟21维只改变HARL接口，不影响任何物理transition、Reward、OPF、MEF或cache结果。

### Observation层判断

**基本通过但非完全惰性。** 原始padding始终为0且稳定；`LayerNorm(288)`会将其变成非零并产生可训练参数梯度，没有NaN/inf，但应在未来mask/异构设计中处理。

### 算法层判断

**未通过。** 虚拟维显著影响old/new log-prob、ratio、clip、entropy、KL、gradient norm、optimizer update和共享trunk。

### 综合选择

**4. 未通过，虚拟维显著干扰BESS学习。**

这不是物理环境错误，也不是bounded distribution错误；问题位于“为了同形接口而新增的维度，被算法当作真实随机控制变量共同优化”。

---

## L. 最小修正建议

### L.1 短训练前必须修正

实施方案B的正式有效动作mask：

```text
IDC mask  = [1, 1, ..., 1]  # 22维
BESS mask = [1, 0, ..., 0]  # 仅第0维
```

并保证同一个mask同时作用于：

- new/old log-prob聚合；
- PPO ratio；
- clip判断；
- policy surrogate；
- entropy；
- KL和训练诊断。

实施后至少重新验证：

1. 物理不变性仍为0；
2. BESS虚拟mean/log-std梯度为0；
3. BESS联合ratio等于物理第0维ratio；
4. clip disagreement为0；
5. shared trunk梯度只来自物理第0维；
6. IDC 22维行为不变；
7. MAPPO smoke和唯一一次1-update通过。

### L.2 MAPPO短训练后修正

- metadata增加 `padding_strategy_version=padding-v1`、真实/padded/virtual维数；
- 日志长期保留effective/full对比；
- 评估是否对BESS observation只归一化真实164维；
- 为未来1维动作头准备checkpoint迁移工具。

### L.3 HAPPO前必须修正

HAPPO与MAPPO复用同样的22维log-prob聚合。必须在HAPPO smoke前让HAPPO factor、ratio、entropy和mask语义完全一致，不能只修MAPPO而留下HAPPO的联合虚拟ratio。

### L.4 建议保持不动

- 底层环境和BESS物理动作映射；
- `HarlPaddedBridge`当前切片行为；
- bounded affine-tanh分布及Jacobian公式；
- Reward、GAE、centralized critic、Grid/Safe；
- 现有24步物理不变性测试。

本报告完成后不进入第八部分，等待确认。

