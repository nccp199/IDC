# 第二阶段实现报告：动态电网状态进入 Centralized State 与 HGTA Bus Nodes

日期：2026-08-13

阶段起点：`49c8aa0555522da71b1c8dae2b34e2d525e410bf`

目标 commit：`feat: add dynamic grid states to centralized HGTA critic`

## 1. 修改前数据流与实现审计

修改前的数据流是：

```text
pandapower.runopp
  -> res_bus / res_line / res_trafo
  -> OPFResult（只保留部分完整数组）
  -> GridCoupledEnv（IDC bus LMP + 全网 min/max/loss 等聚合量）
  -> 288 维 wrapped observation + 6 维 supplemental state
  -> 294 维 centralized state
  -> MLP critic，或 GraphBuilder + HGTA
```

审计得到：

- 当前 AC OPF 的 `res_bus` 实际存在 `vm_pu`、`va_degree`、`p_mw`、`q_mvar`、`lam_p`、`lam_q`。
- `res_line.loading_percent` 是完整的 15 元素结果；solver 原本已保留该数组，但环境主要暴露聚合最大值。
- `res_trafo.loading_percent` 是稳定可得的完整 5 元素结果；solver 原本没有放入 `OPFResult`。
- 修改前完整的逐 bus `p_mw/q_mvar/lam_q`、逐 transformer loading 未进入 info；逐 bus LMP 只在 solver 内存在，环境仅保留 IDC 所在 bus 的聚合 LMP。
- OPF 失败可能留下空字典或非有限的部分结果，因此不能直接拼接进 centralized state。
- 原 294 维 centralized state = `state[0:288]` wrapped observation + `state[288:294]` BESS/功率 supplemental state。
- 原 GraphBuilder 从 centralized state 重构 computing/energy/global 节点，但 bus node 只复制 schema 中的 7 维静态模板；动态电网量没有按 bus 映射。
- 现有 HARL MLP 使用 feature normalization；GraphBuilder 对若干既有物理量自行按固定 reference 缩放。新增动态量在进入 shared state 前已经按固定物理 reference 归一化，不使用测试集或未来 episode 统计。

## 2. 修改后数据流

```text
当前 step AC OPF
  -> 完整逐 bus vm / P / Q / lam_p / lam_q
  -> 完整逐 line / transformer loading
  -> 按真实拓扑计算每个 bus 的 incident max loading
  -> 固定归一化、裁剪与 clip/fallback 诊断
  -> 70 维 grid_bus_dynamic_state
  -> 288 + 6 + 70 = 364 维 centralized state
      -> MAPPO/HAPPO MLP critic（flat）
      -> 同一个 state -> GraphBuilder -> 对应 14 个 bus node -> HGTA critic
```

IDC/BESS actor 继续只读取原 actor observation；没有获得完整逐 bus 状态。

## 3. Bus feature 表

| feature | source | normalization / clip | physical meaning | type |
|---|---|---|---|---|
| nominal voltage | IEEE-14 `net.bus.vn_kv` | 除以全网最大 nominal kV | 静态电压等级 | static |
| base active load | IEEE-14 `net.load.p_mw` | 除以 base-load 最大绝对值 | 静态基准有功负荷 | static |
| base reactive load | IEEE-14 `net.load.q_mvar` | 除以 base-load 最大绝对值 | 静态基准无功负荷 | static |
| has generator | `net.gen.bus` | binary | 是否有可控发电机 | static |
| has external grid | `net.ext_grid.bus` | binary | 是否为外部电网 bus | static |
| has IDC | configured IDC bus 8 | binary | IDC/BESS/PV 接入标记 | static |
| bus index | canonical index 0..13 | `/ 13` | bus 位置标识 | static |
| current voltage deviation | `res_bus.vm_pu` | `(vm_pu - 1.0) / 0.10`，clip `[-2, 2]` | 当前电压相对 1 pu 的偏差 | current |
| current net active power | `res_bus.p_mw` | `/ base_mva`，clip `[-5, 5]` | 当前 bus 净有功状态 | current |
| current net reactive power | `res_bus.q_mvar` | `/ base_mva`，clip `[-5, 5]` | 当前 bus 净无功状态 | current |
| current nodal LMP | `res_bus.lam_p` | `/ grid_lmp_ref`，clip `[-10, 10]` | 当前节点有功边际价格 | current |
| incident max branch loading | 相连 line/trafo 的 `loading_percent` | `/ 100`，clip `[0, 5]` | bus 邻域中最紧张支路负载率 | current |

固定 reference 的来源：IEEE-14 `base_mva=100` 用于 P/Q，既有 `grid_lmp_ref=100` 用于 LMP，电压偏差 reference 为 0.10 pu，支路 loading 原始单位为额定值百分比。全部常数和 clip bounds 在 `grid_model/grid_dynamic_state.py` 中显式定义，并通过 info 输出 metadata。

Pandapower `res_bus` 使用 consumer sign convention：`p_mw/q_mvar > 0` 表示该 bus 净吸收（负荷），`< 0` 表示净注入（发电/外送）。本实现原样保留该符号，再除以 system base；没有为了“直观”而翻转符号。

## 4. Centralized state

- 旧维度：294。
- 新维度：364。
- `state[0:288]`：原 wrapped observation，不变。
- `state[288:294]`：原 supplemental BESS/功率字段，不变。
- `state[294:364]`：14 buses × 5 current dynamic features，bus-major 排列。
- MAPPO+MLP、HAPPO+MLP、HAPPO+HGTA 都从相同的 364 维 share observation space 接收 state。
- actor observation 维度保持 IDC 288、原生 BESS 164；HARL padding 后两者均为 288。actor 网络参数不变。

## 5. OPF solver 与 failure fallback

`OPFResult` 新增并缓存复制：

- `res_bus.lam_q`
- `res_bus.p_mw`
- `res_bus.q_mvar`
- `res_trafo.loading_percent`
- transformer loading limits

已有 `vm_pu`、`lam_p`、line loading 继续使用。GridCoupledEnv 的 info 现在同时保留逐 bus、逐 line、逐 transformer 的 raw current arrays，并输出 transformer 聚合 violation 诊断；reward 和 Safe RL 开关/权重未改。

Failure 策略是无历史依赖的 deterministic neutral fallback：`vm_pu=1`，P/Q/LMP/lam_q/line/transformer/incident loading 全为 0，因此归一化后的 70 维 state 全为 0。缺字段或非有限值按相同的逐字段 neutral value 填充，并记录 missing count/fallback flag。不会复用上一 episode 残值，不会产生 NaN/Inf；checkpoint 仍显式保存当前动态 state 和 raw operating state。

## 6. HGTA 变化与不变量

允许且实际发生的变化：

- bus raw feature：7 -> 12。
- GraphBuilder 从 `state[294:364]` reshape 为 `[batch, 14, 5]`，只拼到同 index 的 bus。
- bus type input projection 的输入宽度随 schema 从 7 变为 12。

保持不变：

- node count = 39。
- edge count = 90。
- relation type、line/transformer/server/IDC relation schema 不变。
- hidden dim = 32、4 heads、2 layers 不变。
- relation-specific attention、Q/K/V、residual、layer norm、pooling、global connection、forecast branch、value head架构不变。

Graph schema/builder 版本更新为 `hgta_graph_v2_dynamic_bus` / `hgta_builder_v2_dynamic_bus`；HGTA architecture version 仍为 `hgta_critic_v1`。

## 7. MLP 与 HGTA 信息公平性

`IDCGridMultiAgentEnv` 只构建一次 364 维 centralized state，HARL bridge 将完全相同的 state 复制到两个 agent 的 share-observation slot。Critic factory 的 MLP 与 HGTA 输入空间都严格要求 `(364,)`：

- MLP：flat 364 state -> existing MLP critic。
- HGTA：same flat 364 state -> deterministic GraphBuilder -> HGTA critic。

因此 HGTA 的差异只在结构化归纳偏置，不包含额外逐 bus 信息。

## 8. 参数量

参数统计来自各方法 Phase 1/Phase 2 final model files 中 state-dict tensor 元素总数：

| method | actor 0 | actor 1 | critic before | critic after | critic delta | total after |
|---|---:|---:|---:|---:|---:|---:|
| MAPPO+MLP | 11,800 | 11,800 | 11,245 | 13,625 | +2,380 | 37,225 |
| HAPPO+MLP | 11,800 | 11,800 | 11,245 | 13,625 | +2,380 | 37,225 |
| HAPPO+HGTA | 11,800 | 11,800 | 102,673 | 102,833 | +160 | 126,433 |

MLP 增量包括 70 个 feature-normalization weight/bias（140）和首层 `70 × 32` 权重（2,240）；HGTA 增量仅为 bus projection 的 `5 × 32=160` 权重。actor 参数完全不变，未做 parameter matching。

## 9. 测试结果

| requirement | result | evidence |
|---|---|---|
| Test 1 bus mapping | PASS | 人工 14×5 唯一值逐 bus 精确映射；GraphBuilder dynamic slice 精确相等 |
| Test 2 physical sensitivity | PASS | 两个真实 AC OPF 工况下 vm、P、Q、LMP、incident loading 均发生有限非零变化 |
| Test 3 topology mapping | PASS | incident max 只遍历真实 15 lines 与 5 transformers 的端点；不是复制全网 max |
| Test 4 no future leakage | PASS | Phase 1 fixed-forecast future-task mutation 回归通过；新增值只由当前 OPFResult 构建 |
| Test 5 centralized parity | PASS | MLP/HGTA 均严格使用同一个 364 state contract |
| Test 6 actor invariance | PASS | IDC 288、BESS 164；动态 slice 变化时 actor observations byte-identical |
| Test 7 GraphBuilder | PASS | 39 nodes、90 edges、bus dim 12、输出有限 |
| Test 8 HGTA backward | PASS | forward/train/backward、value/loss/gradient finite |
| Test 9 OPF failure | PASS | 70 维、全有限、确定性、neutral all-zero normalized fallback |
| Test 10 checkpoint resume | PASS | 15/15 artifact tests；continuous update 2 与 1+resume update 2 的模型、optimizer、RNG、环境（含 dynamic state）及 trajectory 精确一致 |
| Test 11 three smokes | PASS | 三方法各 24 steps / 1 update，24/24 OPF success，reward/loss/action finite，final checkpoint 存在 |

主要命令结果：

- 动态状态/HGTA/causality/logging targeted suite：`40 passed`。
- checkpoint/resume artifact suite：`15 passed`。
- actor/multi-agent/bridge shape regression：首轮发现旧测试索引后修正，复验 `6 passed`；此前完整组除该旧索引外为 `16 passed + 7 subtests`。
- logging + fixed evaluation schema：`14 passed`；fixed evaluation `status=completed, failure_count=0`。
- `git diff --check`：PASS。

三种 smoke 逐项：

| method | reward sum | value loss | IDC policy loss | BESS policy loss | OPF | clip/fallback | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| MAPPO+MLP | -15.56374665 | 27.89411354 | -1.788e-7 | -2.633e-7 | 24/24 | 0 / 0 | PASS |
| HAPPO+MLP | -9.23847170 | 10.09257221 | 1.974e-7 | -0.13085313 | 24/24 | 0 / 0 | PASS |
| HAPPO+HGTA | -10.78282749 | 12.77650833 | -4.371e-7 | -0.17788225 | 24/24 | 0 / 0 | PASS |

Fixed evaluation 的 electrical artifact 已实际写出：bus vm/P/Q/LMP/lam_q/incident loading 均为 `24×14`，line loading 为 `24×15`，transformer loading 为 `24×5`，OPF success 与 network loss 为 24 步序列。episode CSV 继续保留 min/max voltage、line/transformer violation、loss、OPF success 等摘要。

## 10. 未解决问题与明确限制

- line/transformer edge 仍没有原生 dynamic edge attributes；当前使用 incident max loading 作为 node-side 折中。
- line impedance、变压器参数没有进入 attention Q/K/V。
- global node 仍保留 IDC LMP、全网 min/max/loss 等聚合量，与 bus node 存在有意的少量重复；本阶段未删除。
- `lam_q` 被完整提取并保留到 evaluation artifact，但第一版 5-feature centralized state 未使用它。
- fallback 没有单独增加 success-mask feature；现有 global state 已含 `grid_opf_success`，dynamic state 使用 neutral zeros 并另有 info flag。
- normalization references 是固定物理 reference，不会自适应极端分布；因此保留了 clip/missing/fallback frequency 诊断。当前三次 smoke 均无 clip/fallback。
- transformer violation 仅新增为诊断和 evaluation 指标；为遵守范围，未改变现有 `grid_security_penalty`、reward、Safe RL 行为。
- 本阶段没有正式长训练，也没有恢复 grid reward 或 Safe RL。

## 11. Git 与文件边界

- HEAD before：`49c8aa0555522da71b1c8dae2b34e2d525e410bf`。
- commit after：本报告与实现位于同一个 `feat: add dynamic grid states to centralized HGTA critic` commit；由于 commit 不能在自身内容中稳定嵌入自己的 hash，精确 hash 以 `git log -1` 和最终交付信息为准。
- 独立 commit message：`feat: add dynamic grid states to centralized HGTA critic`。
- 用户原有的 `.gitignore`、MEF cache/config、diagnostics/spec 等无关 working-tree 修改不会被覆盖；最终 `git status` 将在提交后再次记录。

主要 Phase 2 文件范围：OPF/result/cache/metrics、`grid_dynamic_state.py`、GridCoupledEnv、central state specs/builders/bridges、MLP/HGTA critic input contract、HGTA graph schema/builder、checkpoint/evaluation/logging、训练入口维度契约和对应测试。本报告本身也纳入该独立 commit。
