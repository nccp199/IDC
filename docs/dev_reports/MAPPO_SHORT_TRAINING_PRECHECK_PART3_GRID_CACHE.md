# MAPPO短训练前检查——第三部分：OPF、LMP、MEF缓存

## 审计结论摘要

本次检查结论选择：

> **3. 未通过，缓存存在正确性风险。**

正式MAPPO环境确实使用独立的内存LRU缓存；隔离、reset生命周期、相同状态命中、失败结果拒绝写入、对象复制保护和性能收益均通过源码与轻量实测。完全相同的第二个24步episode将真实pandapower求解从100次降至0次，耗时由约25秒降至0.017秒。

但当前OPF和MEF共用`0.1 MW`净负荷分箱，而且缓存保存的是**bin中第一个未量化输入的结果**。在`clamp_minus_load=True`的MEF下，零负荷附近存在非线性边界：抽样中把`0 MW`的缓存结果复用于同bin的`0.04 MW`时，`MEF minus`最大绝对误差达到`126.2627604493 kg/MWh`，相对误差`100%`。该误差没有改变本次样本中的OPF成功、MEF成功、电压越限或线路越限，且当前grid reward关闭，因此不改变reward；但MEF是8维grid observation的一部分，会使MAPPO看到与精确电网计算不一致的状态。

在修正或明确关闭MEF近似缓存前，不建议把当前缓存配置用于需要科学可信度的MAPPO短训练。可以继续把它用于管线/性能诊断。

## A. 缓存调用链

```text
train/train_harl_mappo_short.py
  run_training() / prepare_training_environment()
→ marl/envs/harl_env_factory.py
  make_harl_train_env() / make_harl_single_env()
→ train/train_ppo_ultimate.py
  make_unmonitored_env() / make_single_env()
→ env_wrappers/grid_coupled_env.py
  GridCoupledEnv.__init__()
→ grid_model/grid_cache.py
  GridResultCache.__init__()

每次reset/step：
GridCoupledEnv._run_grid_update()
├─ _solve_opf_with_cache()
│  ├─ GridResultCache.make_opf_key()
│  ├─ GridResultCache.get_opf()
│  └─ miss → grid_model.opf_solver.solve_opf()
│             → pandapower.runopp()
│             → OPFResult（含全bus LMP、电压、线路、损耗、dispatch）
│          → GridResultCache.put_opf()
└─ _calculate_mef_with_cache()
   ├─ GridResultCache.make_mef_key()
   ├─ GridResultCache.get_mef()
   └─ miss → GridCoupledEnv._calculate_uncached_mef()
              → grid_model.mef_calculator.calculate_nodal_mef()
                 ├─ base AC-OPF
                 ├─ plus AC-OPF
                 └─ minus AC-OPF
             → GridResultCache.put_mef()
```

| 层级 | 文件与证据 | 类/函数 | 作用 |
| --- | --- | --- | --- |
| 正式入口 | `train/train_harl_mappo_short.py:490-518` | `run_training()` | 调用正式环境准备流程，不覆盖cache配置 |
| probe/train构造 | `train/train_harl_mappo_short.py:427-460` | `prepare_training_environment()` | probe关闭后再次调用相同正式工厂；两次环境不共享对象 |
| HARL工厂 | `marl/envs/harl_env_factory.py:17-40,54-73,117-125` | `make_harl_single_env()`、`make_harl_train_env()` | 每个VecEnv调用一次完整物理环境构造 |
| 原环境工厂 | `train/train_ppo_ultimate.py:65-99,130-145` | `make_single_env()`、`make_unmonitored_env()` | 新建`IDCPriceEnv20D`和`GridCoupledEnv`；可选cache参数继续传入 |
| cache实例 | `env_wrappers/grid_coupled_env.py:88-106` | `GridCoupledEnv.__init__()` | 每个wrapper执行`self.grid_cache = GridResultCache(...)` |
| reset生命周期 | `env_wrappers/grid_coupled_env.py:202-214` | `reset()` | 仅当`cache_clear_on_reset=True`时clear；随后执行初始grid update |
| OPF查询/写入 | `env_wrappers/grid_coupled_env.py:276-305` | `_solve_opf_with_cache()` | hit直接返回副本；miss调用`solve_opf()`后尝试写入 |
| MEF查询/写入 | `env_wrappers/grid_coupled_env.py:307-345` | `_calculate_mef_with_cache()` | hit跳过完整MEF计算；miss计算后尝试写入 |
| OPF底层 | `grid_model/opf_solver.py:12-31,64-103` | `solve_opf()`、`_solve_pandapower_opf()` | deepcopy网络、缩放基础负荷、叠加IDC负荷并运行AC/DC OPF |
| LMP提取 | `grid_model/opf_solver.py:187-210` | `_extract_opf_result()` | `res_bus.lam_p`写入`OPFResult.lmp_by_bus`，因此LMP随OPF结果缓存 |
| MEF底层 | `grid_model/mef_calculator.py:12-130` | `calculate_nodal_mef()` | base、plus、minus三次OPF计算边际排放 |
| cache实现 | `grid_model/grid_cache.py:30-178` | `GridResultCache` | OPF/MEF两个`OrderedDict`共用一个cache对象和配置，分别计数、分别LRU淘汰 |

### 调用链结论

- 正式MAPPO入口没有cache CLI或`harl_mappo_short.yaml`覆盖项，也没有绕过/关闭cache的路径。
- OPF和MEF使用同一个`GridResultCache`实例，但存放在两个独立`OrderedDict`中。
- LMP不是独立求解或独立缓存；它是完整`OPFResult`的一部分。
- OPF hit完全跳过wrapper调用的`solve_opf()`和pandapower。
- MEF hit完全跳过`calculate_nodal_mef()`，因此跳过base/plus/minus三次pandapower OPF。
- 首次MEF miss不会复用OPF cache；每个新状态仍是1次主OPF加3次MEF内部OPF，共4次真实求解。

## B. 最终配置表

正式`main`场景的运行时值由实际环境对象读取，而非仅根据配置文本推断。

| 配置项 | 最终值 | 来源 | 是否生效/运行时覆盖 |
| --- | ---: | --- | --- |
| `enable_grid_cache` | `true` | `configs/config_ultimate.py:153-164` | 生效；正式工厂未覆盖 |
| `cache_opf` | `true` | 同上 | 生效 |
| `cache_mef` | `true` | 同上 | 生效 |
| `cache_load_bin_mw` | `0.1 MW` | 同上；`grid_model/grid_cache.py:38`读取 | OPF和MEF共同使用 |
| `cache_load_scale_bin` | `0.005` | 同上；`grid_model/grid_cache.py:39`读取 | 基础电网load scale分箱；不是MEF负荷分箱 |
| `cache_float_digits` | `6` | `grid_model/grid_cache.py:43`内部默认 | 配置文件未显式设置 |
| MEF `delta_p`量化 | 小数6位round | `grid_model/grid_cache.py:87,173-174` | 不使用`0.005`分箱 |
| `cache_max_size` | `50000` | `configs/config_ultimate.py:159` | 分别限制OPF dict和MEF dict；合计最多100000条 |
| 淘汰策略 | LRU | `grid_model/grid_cache.py:93-100,116-123,176-178` | hit/put移动到末尾，超限弹出最旧项 |
| `cache_clear_on_reset` | `false` | `configs/config_ultimate.py:160` | 由`GridCoupledEnv.reset()`读取，条目跨episode保留 |
| `cache_scope` | `per_worker` | `configs/config_ultimate.py:161` | **键存在但cache类没有读取/执行它**；实际隔离来自每个环境自行new cache |
| `cache_failed_results` | `false` | `configs/config_ultimate.py:162` | 正式配置下失败结果不写入 |
| `cache_verbose` | `false` | `configs/config_ultimate.py:163` | 生效 |
| 磁盘持久化 | 无 | `grid_model/grid_cache.py:45-46` | 仅内存`OrderedDict` |
| per-env/per-worker | 每个环境独立 | `GridCoupledEnv.__init__():106` | 实际生效，不由`cache_scope`键驱动 |
| probe/train/eval | 分别构造 | 正式工厂与第二部分probe流程 | 实测cache对象均不同 |

配置覆盖顺序：

```text
grid_model.grid_cache.DEFAULT_GRID_CACHE_CONFIG
→ configs.config_ultimate.GRID_CACHE_CONFIG（grid_cache_config为None时动态读取）
→ 显式传给GridCoupledEnv的grid_cache_config（若存在）
```

证据位于`env_wrappers/grid_coupled_env.py:92-97,615-620`。正式HARL工厂没有传显式cache配置，因此最终使用`config_ultimate.py`值。`configs/harl_mappo_short.yaml`不参与cache配置。

## C. OPF/MEF缓存键表

实际示例：

```text
OPF: ('opf', 'ac', 8, 3, 1.01, 0.5)
MEF: ('mef', 'ac', 8, 3, 1.01, 0.5, 0.1)
```

| key组成 | OPF | MEF | 是否充分 |
| --- | --- | --- | --- |
| 类型前缀 | `opf` | `mef` | 充分区分两个dict/结果类型 |
| OPF模式 | 有，规范化小写 | 有，规范化小写 | 充分 |
| IDC bus index | 有 | 有 | 充分；LMP读取同一bus |
| hour | 有，整数 | 有，整数 | 充分区分相同倍率/净负荷的不同小时；对纯电网方程略冗余但提高隔离 |
| dynamic load scale | 有，`0.005`分箱 | 有，`0.005`分箱 | 当前动态负荷没有遗漏 |
| IDC/BESS/PV后的净bus负荷 | 有，`P_bus_net_kW`非负部分转MW后按`0.1 MW`分箱 | 同左 | 物理路径正确，但分箱对MEF零点边界过粗 |
| `delta_p_mw` | 无需 | 有，6位round | 充分区分当前delta |
| network case/name | 无 | 无 | 当前每环境case固定且cache不共享，因此当前安全；不适合未来运行期换case |
| 发电机成本/限制 | 无 | 无 | 当前网络对象不可变时安全；若运行期修改会产生陈旧结果 |
| 电压/线路限制 | 无 | 无 | 当前固定时安全；运行期修改后安全指标会陈旧 |
| 排放因子 | 无需OPF | 无 | 当前每环境初始化后固定；若运行期修改会复用错误MEF |
| scenario source | 无 | 无 | 每环境source固定且cache隔离；同一环境若运行期换source但hour/scale相同则不会区分 |
| 对象地址 | 无 | 无 | key稳定，不存在地址导致永不命中的问题 |

### 净负荷数据流

`IDCPriceEnv20D`先形成`P_bus_net_kW`，其中已经包含IDC、BESS充放电和PV影响。`GridCoupledEnv.step()`在`env_wrappers/grid_coupled_env.py:226-228`执行：

```python
idc_load_mw = max(P_bus_net_kW, 0.0) / 1000.0
```

因此：

- BESS真实动作与PV通过最终净负荷间接进入key；
- 负净注入被统一映射为0，因为当前物理网侧不建模外送，和求解输入一致；
- BESS虚拟21维在padding层已被丢弃，不进入物理动作，也不进入key。

### 分箱实现性质

`grid_model/grid_cache.py:167-174`使用：

```python
round(value / bin_size) * bin_size
```

缓存不是用量化后的输入重新求解。miss时`GridCoupledEnv`仍将原始`idc_load_mw`和`load_scale_t`传给solver，然后把该原始结果写到量化key。因此结果具有“同bin第一个访问值决定后续返回值”的顺序依赖。

## D. 生命周期和隔离结论

### reset与episode

- 当前`cache_clear_on_reset=false`，同一环境reset后保留缓存。
- 实测第一次reset后OPF/MEF size均为1；第二次reset仍为1，并各增加一次hit。
- 保留的设计目的是跨episode复用同一`hour + load scale bin + net load bin`的电网求解。
- episode ID未进入key，但OPF/MEF方程没有task、queue或episode内部状态；这些状态只有通过最终净bus负荷影响电网。因此当前跨episode复用原则上成立。
- `hour`和dynamic load scale均进入key，避免不同小时/倍率直接混用。
- 风险来自分箱近似，而不是episode污染。

补充语义：`GridResultCache.clear()`在`grid_model/grid_cache.py:157-159`只清空条目，不重置hit/miss计数。如果未来打开`cache_clear_on_reset`，统计将是“累计查询计数 + 当前episode条目”，需要在日志解释或修正；当前配置不触发。

### probe/train

实测：

- probe reset后：OPF/MEF各1 miss、size各1；
- fresh train环境：hit/miss均0、size均0、`last_info=None`；
- 两者`GridResultCache`不是同一对象。

probe缓存不会传入train。

### train/eval与独立环境

- 同seed创建的两个正式train环境和一个eval环境共有3个不同cache对象。
- 环境1 reset写入后，环境2统计与size完全不变。
- train与eval不会共享cache。

### 未来worker

- 当前正式HARL工厂只支持一个`ShareDummyVecEnv`线程。
- `make_single_env()`每次构造新的`GridCoupledEnv`和cache；未来每个Subproc worker若继续调用该工厂，会在各自进程自然拥有独立cache。
- `cache_scope='per_worker'`本身没有执行逻辑或校验，隔离依赖构造方式。未来重构工厂时必须保持“每个worker新建wrapper”的不变量。

## E. 正确性测试结果

本次没有向仓库新增测试代码；所有验证均为临时PowerShell内联Python脚本，直接使用正式环境工厂或真实`GridCoupledEnv`。没有运行MAPPO网络。

### E1. 命中、隔离、失败和虚拟动作

| 测试 | 首次/再次行为 | solver调用 | cache size | 结果 |
| --- | --- | ---: | ---: | --- |
| 相同OPF状态 | miss → hit | wrapper `solve_opf` 1次 | OPF 1 | 通过 |
| 相同MEF状态 | miss → hit | `calculate_nodal_mef` 1次 | MEF 1 | 通过 |
| OPF缓存结果一致 | LMP、dispatch、电压、线路、损耗一致 | — | — | 通过 |
| MEF缓存结果一致 | plus/minus及排放字段一致 | — | — | 通过 |
| 返回对象修改隔离 | 清空返回副本LMP后重新get，缓存原值仍在 | — | 1 | 通过 |
| hour变化 | key变化 | — | — | miss条件成立 |
| load scale跨bin | key变化 | — | — | miss条件成立 |
| 净负荷跨bin | key变化 | — | — | miss条件成立 |
| bus变化 | key变化 | — | — | miss条件成立 |
| OPF mode变化 | key变化 | — | — | miss条件成立 |
| MEF delta变化 | key变化 | — | — | miss条件成立 |
| OPF失败重试 | miss → 再次miss | 2次 | 0 | 通过 |
| MEF失败重试 | miss → 再次miss | 2次 | 0 | 通过 |
| MEF base OPF失败 | 立即返回失败 | 1次内部OPF | 不写cache | 通过 |
| MEF plus OPF失败 | 返回失败 | 2次内部OPF | 不写cache | 通过 |
| MEF minus OPF失败 | 返回失败 | 3次内部OPF | 不写cache | 通过 |
| LRU上限（测试容量2） | 插入3条后最旧项被淘汰 | — | 2 | 通过 |
| BESS虚拟21维变化 | 两个正式环境OPF/MEF key和stats相同 | — | 相同 | 通过 |

失败不缓存的正式结论受`cache_failed_results=false`约束。实现允许显式改成`true`；因此“所有可能配置下绝不会缓存失败”不成立，但**当前正式配置**下成立。

### E2. Cache on/off 24步物理对比

相同seed、相同初始task RNG状态、相同24步合法动作序列下，cache关闭与cache开启第一个episode比较：

| 字段 | 24步最大绝对差异 |
| --- | ---: |
| reward | 0 |
| SOC、充电功率、放电功率 | 0 |
| IDC功率、grid功率、bus净功率 | 0 |
| energy、cost、carbon | 0 |
| OPF/MEF success | 完全相同 |
| LMP | 0 |
| MEF plus/minus | 0 |
| min/max voltage | 0 |
| max line loading | 0 |
| network loss | 0 |
| completion、backlog | 0 |
| done/hour | 完全相同 |
| grid reward penalty | 0 |

原因是第一个episode的25个状态（reset + 24 step）全部为miss，cache没有复用近似bin结果。

相同环境中恢复task RNG初始状态后运行完全相同的第二个episode，以上字段最大差异仍全部为0；25个OPF和25个MEF查询全部命中。

## F. 近似误差结果

### 方法

- 从真实24步动作采样轨迹选择8个代表状态：低/中/高净负荷、最低/最高基础负荷倍率、BESS充电、BESS放电和PV高发。
- 对每个真实状态选择同一`0.1 MW` cache bin内、相差`0.04 MW`的另一状态。
- 分别执行不使用缓存的真实AC-OPF与MEF。
- 把第一个真实结果视为当前“bin首个结果”，与第二个状态的真实结果比较。
- 样本数为8；部分物理类别会落在同一小时/状态，但均按类别保留并计入汇总。

| 指标 | 最大绝对误差 | 平均绝对误差 | 最大相对误差 | 平均相对误差 |
| --- | ---: | ---: | ---: | ---: |
| LMP | 0.0017137594 | 0.0010912868 | 0.004314% | 0.002731% |
| MEF plus (kg/MWh) | 1.1095530158 | 0.2925867206 | 0.394353% | 0.105178% |
| MEF minus (kg/MWh) | **126.2627604493** | **32.2026074022** | **100%** | **37.595774%** |
| min voltage (p.u.) | 0.0000048591 | 0.0000030227 | 0.000479% | 0.000298% |
| max voltage (p.u.) | 0.0000144607 | 0.0000106357 | 0.001337% | 0.000982% |
| max line loading (%) | 0.0000356582 | 0.0000210088 | 0.002943% | 0.001719% |
| network loss (MW) | 0.0006233702 | 0.0004038120 | 0.006890% | 0.004428% |
| grid reward | 0 | 0 | 0 | 0 |

### 安全判断翻转

在8个样本中：

- OPF成功/失败翻转：0；
- MEF成功/失败翻转：0；
- 电压越限判断翻转：0；
- 线路越限判断翻转：0；
- grid reward差异：0（当前grid reward关闭）。

### MEF minus边界原因

`grid_model/mef_calculator.py:74-106`先计算：

```text
minus_load = max(base_load - delta_p, 0)
MEF_minus = (base_emission - minus_emission) / delta_p
```

当前`delta_p=0.1 MW`，而负荷cache bin也是`0.1 MW`。当bin首值为`0 MW`时，base和minus均为0，MEF minus为0；同bin中的`0.04 MW`仍把minus clamp到0，但base emission已经变化，真实MEF minus非0。直接复用零负荷结果产生最大`126.26 kg/MWh`误差。

该误差归一化进入grid observation时约为`0.1263`（当前`grid_mef_ref=1000`），不是可以忽略的机器浮点误差。

## G. 性能结果

计数通过测试期monkeypatch记录：

- `wrapper OPF`：GridCoupledEnv主OPF调用；
- `MEF calls`：MEF计算次数；
- `MEF内部OPF`：每次MEF的base/plus/minus真实求解。

| 模式 | 总耗时（含reset+24步） | 主OPF调用 | MEF调用 | MEF内部OPF | 真实OPF总数 | 本episode hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache关闭 | 25.0971s | 25 | 25 | 75 | 100 | 0% |
| cache开启，第1个episode | 25.2766s | 25 | 25 | 75 | 100 | 0% |
| cache开启，第2个完全相同episode | 0.0173s | 0 | 0 | 0 | 0 | 100% |
| cache开启，自然连续第1个episode（独立测量） | 26.7866s | 25 | 25 | 75 | 100 | 0% |
| cache开启，自然连续第2个episode | 0.0188s | 0 | 0 | 0 | 0 | 100% |

按24个step粗略计算，前三种模式单步平均耗时约为：

- cache关闭：`1.0457s/step`；
- cache开启首次：`1.0532s/step`；
- 完全命中：`0.00072s/step`。

第二个自然episode没有恢复task RNG，但在本固定动作抽样下，其每小时净负荷仍落入第一个episode相同bin，因此25个OPF和25个MEF全部命中。累计统计显示hit rate为50%，这是两episode合计；第二episode增量hit rate是100%。

性能收益非常明确，但第一个新状态episode不会提速，且收益来自近似bin复用。

## H. 问题清单

### 阻塞MAPPO短训练

1. **MEF minus在零负荷/clamp边界存在显著同bin复用误差。**
   - 证据：`0 → 0.04 MW`同key时最大误差`126.2627604493 kg/MWh`，相对误差100%。
   - 当前reward不受影响，但MAPPO local observations包含MEF，会改变策略输入。
   - 对“管线能否运行”的smoke不构成阻塞；对“短训练结果是否代表精确物理反馈”构成阻塞。

### 必须在正式长训练前修正

1. OPF与MEF共用同一`cache_load_bin_mw`，无法针对MEF的`delta_p`/clamp非线性单独设定精度。
2. cache key不包含network/case签名、发电机约束、线路/电压约束或排放因子。当前per-env不可变假设下安全，但未来运行期切换case/参数会得到陈旧结果。
3. `cache_scope='per_worker'`只是未读取的声明键；未来多worker正确性依赖工厂仍为每个worker构造新环境。

### 建议改进

1. `stats()`和step info没有eviction count与actual solver calls，无法仅靠正式日志核对hit统计和真实求解次数。
2. `cache_max_size=50000`分别作用于OPF和MEF dict，而非总容量；配置名容易被理解成总共50000。
3. `clear()`不重置统计；若未来启用reset清理，需要明确累计统计语义。
4. Python `round()`在bin半边界采用banker's rounding；这是确定性的，但边界两侧会跳bin，建议用测试固定并文档化。

### 可以保持现状

1. 每个环境独立cache对象，probe/train/eval隔离正确。
2. hour、dynamic load scale、bus、OPF mode和delta均进入对应key。
3. LMP随完整OPF结果缓存，命中bus正确。
4. hit返回深度重建的轻量dataclass和新dict；调用方修改不会污染缓存。
5. 当前正式配置的OPF/MEF失败结果不缓存，并会在相同状态再次求解。
6. LRU淘汰基本行为正确。
7. reset保留cache符合跨episode复用设计。
8. BESS虚拟21维不进入cache key。

## I. 第三部分判断

选择：

> **3. 未通过，缓存存在正确性风险。**

判断边界：

- 缓存基础实现和生命周期不是问题；
- OPF/LMP近似误差在本次8样本中很小，未出现安全判断翻转；
- MEF plus误差最高约0.394%；
- MEF minus在零负荷边界的100%相对误差会直接进入策略观测，因此不能把当前配置视为“缓存不改变MAPPO看到的物理状态”。

如果下一步只是再次执行smoke或训练流程诊断，可以保留缓存；如果下一步要产生可解释的MAPPO短训练结果，应先处理MEF cache精度问题或暂时关闭MEF cache。

## J. 最小修正建议

本次未实施任何修正。建议等待确认后只选择以下最小方案之一。

### 推荐正式方案

```text
修改：
- grid_model/grid_cache.py
  - 为MEF增加独立的负荷key精度，例如cache_mef_load_bin_mw；
  - 对base_idc_load_mw < delta_p_mw的clamp区域使用更细分箱或精确key；
  - 保持OPF现有0.1 MW分箱不变。

- configs/config_ultimate.py
  - 显式配置MEF独立分箱值；
  - 不改变OPF、MEF算法和物理输入。

新增：
- marl/tests/test_grid_cache_formal_entry.py
  - 固化隔离、失败不缓存、命中、reset、零负荷MEF边界、on/off等价和性能计数测试。
```

### 短训练前的保守配置方案

```text
仅配置：
- 暂时cache_mef=false，保留cache_opf=true；或
- 短训练暂时完全关闭grid cache，获得精确基线。
```

关闭MEF cache会保留主OPF缓存，但每个状态仍需3次MEF内部OPF，性能收益会显著下降。完全关闭cache则本次测得约25秒/24步，不影响正确性但短训练耗时更长。

不建议仅通过扩大分箱或忽略MEF minus来追求命中率。

## K. 当前step info中的缓存统计字段

产生位置：`env_wrappers/grid_coupled_env.py:526-544`。

| 字段 | 是否存在 | 含义 |
| --- | --- | --- |
| `grid_cache_enabled` | 是 | 总cache开关 |
| `grid_opf_cache_enabled` | 是 | OPF cache有效开关 |
| `grid_mef_cache_enabled` | 是 | MEF cache有效开关 |
| `grid_opf_cache_hit` | 是 | 当前grid update的主OPF是否hit |
| `grid_mef_cache_hit` | 是 | 当前grid update的MEF是否hit |
| `grid_opf_cache_hit_count` | 是 | 环境生命周期累计OPF hits |
| `grid_opf_cache_miss_count` | 是 | 环境生命周期累计OPF misses |
| `grid_opf_cache_hit_rate` | 是 | 累计hits/(hits+misses) |
| `grid_mef_cache_hit_count` | 是 | 环境生命周期累计MEF hits |
| `grid_mef_cache_miss_count` | 是 | 环境生命周期累计MEF misses |
| `grid_mef_cache_hit_rate` | 是 | 累计hits/(hits+misses) |
| `grid_cache_opf_size` | 是 | 当前OPF条目数 |
| `grid_cache_mef_size` | 是 | 当前MEF条目数 |
| `grid_cache_load_bin_mw` | 是 | 两类cache共用的净负荷bin |
| `grid_cache_load_scale_bin` | 是 | 基础负荷倍率bin |
| eviction count | 否 | LRU淘汰未计数 |
| actual wrapper OPF calls | 否 | 本次只能通过测试patch计数 |
| MEF内部OPF calls | 否 | 正式info/logger未记录 |
| `cache_max_size` | stats中有，step info无 | 单个dict最大条目数 |
| `cache_failed_results` | stats中有，step info无 | 是否允许缓存失败结果 |

## L. 关键函数微分析与不变量

本节按照逐函数、输入、状态写入和跨函数依赖记录关键证据。

### `GridResultCache.make_opf_key()` / `make_mef_key()`

**Purpose：** 两个函数把影响当前电网求解的运行参数规范化为可哈希tuple。它们决定哪些物理状态被认为等价，因此直接控制命中率和近似误差边界。

**Inputs & Assumptions：** 输入来自同一个`GridCoupledEnv`实例；假定network、约束和排放因子在实例生命周期内不变；假定hour、load scale和净负荷足以表达所有动态电网状态；假定MW单位已经由wrapper正确转换；假定`delta_p`为有限稳定值。

**Outputs & Effects：** 返回不含对象地址的tuple；不写cache；负荷和倍率经过分箱；mode规范化；bus/hour强制整数；MEF额外记录delta。

**Block-by-block：**

- `grid_model/grid_cache.py:61-68`构造OPF key。模式、bus、hour先保留离散身份，倍率和IDC负荷最后量化；这是因为命中必须先满足拓扑位置与时间状态一致。
- `grid_model/grid_cache.py:80-88`构造MEF key。MEF的base状态与OPF相同，再附加delta；为什么必须附加delta：有限差分步长改变plus/minus负荷与分母，省略会直接混用不同数学问题。
- `grid_model/grid_cache.py:167-174`使用nearest-bin round而非量化求解。第一性原理：cache正确复用要求“同key输出可接受地等价”；当前MEF clamp边界违反连续近似假设。

**不变量：** 同样的规范化输入产生同key；不同类型永不共用key；mode/bus/hour/delta变化产生不同身份；同bin值有意共享key。

**Cross-function dependencies：** 调用方是wrapper两个with-cache函数；量化结果决定`get_*`和`put_*`的地址；solver实际收到原值而非key中的量化值；dynamic scenario由`_grid_scenario_values()`提供hour和scale。

### `GridResultCache.get_*()` / `put_*()` / `_trim()`

**Purpose：** 查询函数维护hit/miss与LRU顺序并返回隔离副本；写入函数执行开关、容量和失败策略后保存副本；trim保证单个dict容量上限。

**Inputs & Assumptions：** key由稳定builder生成；result是`OPFResult`或`MEFResult`；`success`真实代表可复用性；容量非负；下游只需要dataclass字段而不依赖`raw_result`。

**Outputs & Effects：** miss增加miss count；hit增加hit count并移动LRU；put可能写入、更新和淘汰；所有保存/返回均重建dataclass；OPF缓存副本将`raw_result`设为None。

**Block-by-block：**

- `grid_model/grid_cache.py:90-101,113-124`先检查开关，再区分hit/miss。为什么hit移动末尾：LRU必须让最近访问条目晚于冷条目淘汰。
- `grid_model/grid_cache.py:103-110,126-134`在写入前拒绝disabled、零容量和当前配置下的失败结果。为什么在copy前拒绝：避免无效对象占内存和污染key。
- `grid_model/grid_cache.py:176-178`循环弹出最旧项。实测容量2插入3项后仅保留key1/key2。
- `grid_model/grid_cache.py:185-235`逐字段复制dict。为什么重建：避免调用方修改缓存中的可变字典；实测清空返回LMP不影响后续get。

**不变量：** 正式配置下失败不增加size；hit不调用solver；返回对象与保存对象不共享dict；每个OPF/MEF dict不超过`cache_max_size`。

**Cross-function dependencies：** `GridCoupledEnv`只根据`get_* is not None`判断hit；step info读取`stats()`；下游metrics、LMP和emission只依赖复制保留的轻量字段，不依赖`raw_result`。

### `GridCoupledEnv._solve_opf_with_cache()`

**Purpose：** 将主OPF查询、真实求解和条件写入组合成单一路径，并向上层返回当前是否hit。它直接决定LMP、电压、线路和损耗是新求解还是复用。

**Inputs & Assumptions：** `idc_load_mw`已非负且单位MW；hour/scale来自当前grid scenario；bus、mode和network在wrapper生命周期内稳定；cache对象属于当前环境；solver返回结构完整的`OPFResult`。

**Outputs & Effects：** 返回`(OPFResult, hit_bool)`；hit只更新cache LRU/计数；miss调用pandapower并可能写cache；disabled时始终真实求解且hit为False。

**Block-by-block：**

- `env_wrappers/grid_coupled_env.py:277-287`只有总开关和OPF开关同时有效才构建key并查询。为什么先query：避免任何solver副作用与成本。
- `env_wrappers/grid_coupled_env.py:288-296`miss时把**原始值**传给solver，随后按量化key写入。为什么产生顺序依赖：key代表bin，但result代表首个原始点。
- `env_wrappers/grid_coupled_env.py:298-305`disabled分支完全绕过cache计数。实测cache关闭时hit/miss均保持0，同时真实求解25次主OPF。

**不变量：** hit时真实solver调用0；主OPF的bus与LMP读取bus一致；失败结果在正式配置下不写入；返回hit flag与当前查询一致。

**Cross-function dependencies：** key来自`GridResultCache.make_opf_key()`；真实结果来自`grid_model.opf_solver.solve_opf()`；下游`extract_grid_metrics()`和`_inject_grid_info()`消费同一结果；`put_opf()`执行最终失败策略。

### `GridCoupledEnv._calculate_mef_with_cache()` / `_calculate_uncached_mef()`

**Purpose：** 为当前bus净负荷缓存完整nodal MEF结果，并在miss时运行base/plus/minus三次OPF。异常被转换为`success=False`的`MEFResult`，使wrapper仍能完成info构造。

**Inputs & Assumptions：** 使用与主OPF相同的base state；delta、emission factors、bus和mode在环境生命周期内固定；`clamp_minus_load=True`；MEF函数的success综合反映三次OPF与有限数值。

**Outputs & Effects：** hit跳过三次OPF；miss调用一次MEF函数（内部最多三次OPF）；成功结果写入MEF dict；异常/失败在正式配置下不写入。

**Block-by-block：**

- `env_wrappers/grid_coupled_env.py:308-319`构造包含delta的key并优先查询。为什么MEF独立于OPF cache：实现把完整MEF作为一个结果缓存，但代价是首次MEF内部base OPF不会复用主OPF。
- `env_wrappers/grid_coupled_env.py:320-324`miss调用uncached函数并尝试写入。量化key与原始输入结果的差异同样造成bin首值依赖。
- `env_wrappers/grid_coupled_env.py:326-345`将异常转成失败dataclass。为什么不抛出：grid wrapper选择通过info暴露失败；启动probe另外对初始失败执行fail-fast。
- `grid_model/mef_calculator.py:39-100`依次执行base、plus、minus；任何一步失败立即返回false。实测对应调用次数分别为1、2、3，且wrapper重复请求仍重算。
- `grid_model/mef_calculator.py:74-106`对minus负荷clamp到0，但仍除以固定delta。这是零点附近MEF非线性和cache近似误差的来源。

**不变量：** MEF hit时三次内部OPF均为0；base/plus/minus任一失败使最终success=False；正式配置不缓存该失败；delta必须进入key；返回bus与当前IDC bus一致。

**Cross-function dependencies：** emission factors来自`GridCoupledEnv.__init__()`；MEF内部直接调用`grid_model.mef_calculator.solve_opf()`，不通过wrapper OPF cache；step observation和info读取plus/minus；cache copy保留所有MEF轻量字段。

## M. 检查范围与未执行事项

- 未修改cache、OPF、MEF、物理环境、reward、Bridge、padding、runner、MAPPO、GAE或配置。
- 未新增仓库测试文件；只新增本Markdown报告。
- 未开始MAPPO 5-update、40-update或任何策略训练。
- 未检查IDC电价。
- 未实现多worker、完整logger、HAPPO、HGTA或Safe RL。
- 分箱误差是8个定向样本，不是全状态空间统计；未观察到翻转不等同于证明所有边界永不翻转。
- 性能数字为本机单次轻量测量，不代表不同硬件的绝对耗时。

本次检查到第三部分为止，等待下一步确认。
