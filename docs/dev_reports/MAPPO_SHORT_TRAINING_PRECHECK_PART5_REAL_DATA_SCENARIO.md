# MAPPO短训练前检查——第五部分：真实数据与场景配置

检查日期：2026-07-28  
项目：`C:\Users\bulio\Desktop\IDC\ultimate_simplify`  
HARL：`C:\Users\bulio\Desktop\IDC\HARL`  
正式入口：`train/train_harl_mappo_short.py`  
正式环境工厂：`marl/envs/harl_env_factory.py`

## 结论摘要

- 当前正式 `main` 是一个**混合工程场景**：NEMS格式数据派生的固定24小时电网负荷倍率 + IEEE14标准测试系统 + 人工或随机合成的IDC电价、碳因子、温度、PV、任务和服务器参数。
- 正式运行时实际读取的项目外部数据文件只有 `data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale.csv`。原始NEMS文件、处理元数据和全日期文件不在环境启动时读取。
- IDC购电价仍是代码内人工分时电价 `[0.35, 0.65, 1.05]` 元/kWh，不是NEMS `USEP`。环境虽然把 `USEP` 读入 `grid_reference_usep`，但它只作为诊断字段，不进入IDC成本、观测中的 `price_t`、OPF或reward。
- 当前NEMS处理方式是：从2026-05-01至2026-05-14的14个完整日中，以日均需求最接近14日日均值选出2026-05-07；每两个30分钟period取算术平均形成1小时；再除以该日24小时需求均值6815.327020833333 MW得到倍率。该倍率保留单日峰谷比例，但环境不使用绝对MW需求。
- 当前单位链在既有混合场景下未发现会阻塞短训练的错误：IDC功率kW乘1小时得到kWh，再乘人工元/kWh；Grid侧将kW除以1000后以MW注入OPF；MEF奖励路径先把kWh除以1000为MWh。Grid reward当前关闭。
- 不能从仓库证明原始文件的官方URL、许可证、时区或下载hash。它具有NEMS/USEP表头和新加坡数据目录语义，但在无外部核验条件下只能称为“仓库内NEMS格式外部数据副本”。
- 正式环境固定动作实测通过：24步、hour 0–23、OPF 24/24成功、MEF 24/24成功、全部审计数值有限、末步正常终止。
- 第五部分判断选择：**2. 基本通过，短训练可用，但正式实验需补充真实数据。** 当前结果适合MAPPO工程短训练和管线基线，不足以支撑“真实数据场景”或“真实新加坡微电网实验”的论文表述。

## A. 数据调用链

### A.1 正式调用链

```text
configs/harl_mappo_short.yaml
  env.experiment_case = main
    ↓
marl/envs/harl_env_factory.py::make_harl_single_env()
    ↓
configs/experiment_cases.py::get_experiment_case("main")
  deepcopy(ENV_CONFIG, REWARD_CONFIG, DATA_CONFIG)
  main不覆盖基础数据配置
    ↓
train/train_ppo_ultimate.py::make_unmonitored_env()
    ↓
train/train_ppo_ultimate.py::make_single_env()
  build_external_series_from_config(DATA_CONFIG, horizon=24)
  + IDC_SCALE_CONFIG
  + server_seed/task_seed
    ↓
data_io/data_loader.py::build_external_series_from_config()
  price/carbon/temperature: path=None → 返回None
  PV: path=None + use_default_pv_curve=True → 正弦日照曲线
  WT: path=None + default_zero=True → 24个0
    ↓
envs/idc_price_env.py::IDCPriceEnv20D.__init__()
  None价格 → IDCEnergyTaskModel.create_price_curve()
  None碳因子 → _create_carbon_factor_curve()
  None温度 → 25 + 5*sin(...)
  reset() → seed控制的随机task
    ↓
env_wrappers/grid_coupled_env.py::GridCoupledEnv.__init__()
  固定GRID_CONFIG / GRID_REWARD_CONFIG / GRID_SCENARIO_CONFIG
  _load_grid_scenario()
    ↓
data/grid_scenarios/nems_singapore/processed/nems_24h_load_scale.csv
  grid_load_scale → IEEE14基础负荷倍率
  usep_sgd_per_mwh → 仅grid_reference_usep诊断字段
```

### A.2 文件与函数表

| 数据类别 | 配置入口 | 加载/生成函数 | 最终进入环境的位置 |
| --- | --- | --- | --- |
| IDC电价 | `configs/config_ultimate.py:68-84`，`price_csv_path=None` | `data_io/data_loader.py:140`返回`price_t=None`；`idc_model/power_model.py:169`生成 | `IDCPriceEnv20D.price_t`；`envs/idc_price_env.py:479,607`用于同一步成本 |
| 碳因子 | `carbon_csv_path=None` | `IDCPriceEnv20D._create_carbon_factor_curve()`，`envs/idc_price_env.py:313` | `carbon_factor_t`；`envs/idc_price_env.py:480,608` |
| 温度 | `temperature_csv_path=None` | `envs/idc_price_env.py:224`的正弦公式 | `T_amb[t]`；同一步PUE/COP与制冷功率 |
| PV | `pv_csv_path=None`、`use_default_pv_curve=True`、500 kW | `data_io/data_loader.py::_build_default_pv_curve()` | `IDCPriceEnv20D.pv_t`；同一步抵扣本地母线净负荷 |
| WT | `wt_csv_path=None` | `load_optional_time_series_csv(..., default_zero=True)` | `wt_t`，当前全0；未进入功率抵扣公式 |
| task | `ENV_CONFIG.num_tasks=30`、`IDC_SCALE_CONFIG.task_workload_scale=100` | `IDCEnergyTaskModel.create_demo_tasks()` / `create_random_tasks()` | `IDCPriceEnv20D.reset()`中的`tasks`与`lambda_t` |
| server | `IDC_SCALE_CONFIG`与模型默认工程范围 | `IDCPowerModel.__init__()`随机采样，seed控制 | `model.P_idle/P_max/C_server` |
| IEEE14 | `GRID_CONFIG.case_name=ieee14` | `grid_model/ieee14_loader.py::load_ieee14_case()` → `pandapower.networks.case14()` | `GridCoupledEnv.grid_case` |
| NEMS负荷倍率 | `GRID_SCENARIO_CONFIG.grid_load_scale_path` | `GridCoupledEnv._load_grid_scenario()` | `solve_opf(..., load_scale=grid_load_scale_t[hour])` |
| NEMS USEP | 同一processed CSV的`usep_sgd_per_mwh` | `GridCoupledEnv._load_grid_scenario()` | 只写`info["grid_reference_usep"]` |
| BESS初态 | `ENV_CONFIG.bess_soc_init=0.50`及固定容量/功率 | `IDCPriceEnv20D.__init__/reset` | 每episode固定SOC 0.5；容量/功率按IDC scale乘100 |

### A.3 配置是否被实际读取

- `get_experiment_case("main")`在`configs/experiment_cases.py:50`对三个基础字典执行`deepcopy`，`main`分支不覆盖任何字段。因此不存在浅拷贝导致的跨运行污染。
- 正式工厂在`marl/envs/harl_env_factory.py:31-39`把case内三个字典交给`make_unmonitored_env()`，数据配置真实进入环境构造链。
- `DATA_CONFIG`中的所有IDC外部CSV路径均为`None`，所以不存在“配置填了真实IDC数据但正式入口没有读取”的情况；当前明确选择的是默认/合成曲线。
- NEMS processed路径位于`GRID_SCENARIO_CONFIG`，不是case字典的一部分，但`train/train_ppo_ultimate.py::make_single_env()`构造`GridCoupledEnv`时始终传入该全局配置，因此正式MAPPO实际读取它。
- 仓库内没有另一套被正式MAPPO继承的真实IDC电价、碳因子、温度或PV CSV。旧诊断脚本与single/multi-IDC场景模块不改变正式`main`数据链。

## B. 真实/合成数据总表

本报告使用以下严格分类：①仓库内外部原始数据副本；②由外部数据清洗/聚合/归一化得到；③代码人工曲线；④seed控制的随机合成；⑤标准测试系统；⑥固定工程参数。只有①和②包含外部实测数据成分；②不能表述为原始真实量。

| 数据 | 严格分类 | 来源文件/生成函数 | 单位 | 分辨率 | 长度 | 正式环境用途 |
| --- | --- | --- | --- | --- | ---: | --- |
| IDC电价 | 代码人工曲线，非真实 | `idc_model/power_model.py::create_price_curve()` | 元/kWh | 1小时 | 24 | 成本、reward、观测 |
| 碳因子 | 代码人工场景假设，非真实 | `IDCPriceEnv20D._create_carbon_factor_curve()` | kgCO2/kWh | 1小时 | 24 | 碳排放、reward、观测 |
| 环境温度 | 代码正弦曲线，非真实 | `25 + 5*sin(pi*(hour-8)/12)` | °C | 1小时 | 24 | PUE/COP、制冷功率、观测 |
| PV可用功率 | 代码正弦日照曲线，非真实 | `data_io/data_loader.py::_build_default_pv_curve()` | kW | 1小时 | 24 | 抵扣本地母线净负荷 |
| WT | 固定工程占位 | `load_optional_time_series_csv(...default_zero=True)` | kW接口 | 1小时 | 24 | 当前全0，未抵扣负荷 |
| IEEE14基础负荷 | 标准测试系统，非新加坡真实电网 | `pandapower.networks.case14()` | MW/Mvar | 静态基准 | 14母线 | OPF基础网络与负荷 |
| NEMS需求原始数据 | 仓库内外部数据副本；官方来源链未在仓库证明 | `raw/USEP_May-2026.csv`的`DEMAND (MW)` | MW | 30分钟period | 672=14×48 | 不直接运行时读取 |
| 电网负荷倍率 | 外部需求聚合并按单日均值归一化 | `processed/nems_24h_load_scale.csv` | 无量纲 | 1小时 | 24 | 缩放IEEE14基础负荷 |
| NEMS价格 | 仓库内外部数据副本及聚合值 | 原始/processed的`USEP` | SGD/MWh | 原始30分钟；processed 1小时 | 672/24 | 只作诊断参考，不是IDC电价 |
| task arrival | seed控制随机合成 | `create_random_tasks()`，到达时刻1–18；另有hour0初始积压 | workload单位 | 1小时 | 30随机任务+1初始任务 | 调度与队列 |
| task workload | profile范围内随机合成，再乘100 | `create_task()` / task profiles | 模型workload单位 | 按任务 | 31任务 | 调度、队列、reward |
| server参数 | seed控制的工程范围随机参数 | `idc_model/power_model.py::__init__()` | W、workload容量 | 环境构造时一次 | 20组 | IDC功耗和处理能力 |
| IEEE14网络 | 标准测试系统 | `pandapower.networks.case14()` | pandapower标准单位 | 静态 | 14母线 | AC OPF/MEF |
| BESS初始状态 | 固定工程参数 | `bess_soc_init=0.5` | SOC无量纲 | 每episode reset | 1 | 初始储能状态 |

因此，`main`不能准确命名为“真实数据场景”。准确表述应是：

> 使用NEMS格式需求数据派生的固定单日电网负荷倍率，在IEEE14测试网中运行的合成IDC/BESS双Agent混合场景。

## C. IDC电价专项结论

### C.1 正式训练实际电价

```text
hour:    0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23
price: 0.35  0.35  0.35  0.35  0.35  0.35  0.35  0.65  0.65  0.65  1.05  1.05  1.05  1.05  1.05  0.65  0.65  0.65  1.05  1.05  1.05  0.65  0.65  0.65
```

- 单位由源码明确写为“元/kWh”。
- 来源是`idc_model/power_model.py:169-204::create_price_curve()`；源码同时明确说明它是仿真设定值，不代表地区真实商业电价。
- `DATA_CONFIG.price_csv_path=None`，未读取CSV。
- 环境实例构造后该数组固定，reset不重新生成；所有episode相同。
- 它与NEMS需求或USEP没有生成关系，只是人工峰谷平分时电价。
- 成本公式为`cost_t = grid_energy_kWh * price_now`，位于`envs/idc_price_env.py:607`，因此当前人工价格的量纲与公式一致。

### C.2 仓库内NEMS价格候选

| 文件 | 表头/内容 | 单位 | 分辨率 | 正式入口用途 |
| --- | --- | --- | --- | --- |
| `data/grid_scenarios/nems_singapore/raw/USEP_May-2026.csv` | `DATE, PERIOD, USEP, LCP, DEMAND, SOLAR, TCL, RUSEP, MAP, MAPT...` | 表头价格为$/MWh；处理脚本将USEP解释为SGD/MWh | 30分钟，14日×48 | 不直接读取 |
| `.../processed/nems_24h_load_scale.csv` | 2026-05-07的`demand_mw, usep_sgd_per_mwh, solar_mw, grid_load_scale` | SGD/MWh、MW、无量纲 | 1小时，24行 | 读取倍率；USEP仅写入info |
| `.../processed/nems_all_days_hourly_profiles.csv` | 14日的小时需求、USEP、按日/全局均值倍率 | SGD/MWh、MW、无量纲 | 1小时，336行 | 当前不读取 |
| `.../processed/nems_24h_load_scale_meta.json` | 选日、聚合和统计元数据 | 混合 | 文件级 | 当前不读取 |

原始样本首行是：`01-May-2026, PERIOD=1, USEP=300.36, LCP=0.00, DEMAND=6777.882 MW, SOLAR=0`。文件没有母线/节点ID，所以不能把USEP或其他价格列称为节点电价。正式Grid LMP来自pandapower IEEE14 OPF，与NEMS USEP是两套不同来源和币种语义的量。

### C.3 作为IDC购电价的接入判断

- processed USEP是157.98–181.735 SGD/MWh，换算后为0.15798–0.181735 SGD/kWh。若直接把157.98–181.735乘以kWh，成本会放大1000倍。
- 原始14日USEP范围103.29–349.95 SGD/MWh，即0.10329–0.34995 SGD/kWh；没有缺失、NaN、0或负值。当前数据没有负价，但正式接入仍应定义对负价、尖峰和缺失值的策略，不能据此假设未来文件不会出现。
- 当前人工价格0.35/0.65/1.05元/kWh与选中日0.15798–0.181735 SGD/kWh不是同币种，也不应只按数值直接比较；人工曲线的峰谷跨度明显更大。
- 当前`price_ref=1.50`适配人工最大值1.05。若切换选中日USEP换算值，其最大归一化值约为0.121；继续使用1.50会把价格观测压到很小区间。
- 有效`cost_ref`在server-group缩放后是`60×100=6000`，真实价格接入后仍需基于实际IDC能量和目标reward量级重新核验；这属于配置标定，不应在数据加载时偷偷clip。
- 最小正式接入方式应是：先在可追溯预处理阶段明确`SGD/MWh → SGD/kWh`除以1000，再通过新的显式data/case配置提供24小时数组，同时记录币种、日期、时区、hash和转换；随后单独复核`price_ref`、`cost_ref`和reward量级。本次未做任何接入或修改。

## D. NEMS负荷处理链

### D.1 原始到运行时

```text
raw/USEP_May-2026.csv
  14个日期（2026-05-01～2026-05-14）
  每日48个PERIOD，DEMAND单位MW，USEP表头$/MWh
    ↓ scripts/build_grid_load_scale_from_nems.py
  检查完整日必须包含period 1..48
  typical = 日均需求最接近14个完整日日均需求的日期
    ↓ selected_date = 2026-05-07
  hour h = mean(period 2h+1, period 2h+2)
  demand/usep/solar均取算术平均，不是求和、最大值或抽样
    ↓
  denominator = 2026-05-07的24小时平均需求
              = 6815.327020833333 MW
  grid_load_scale[h] = hourly_demand_mw[h] / denominator
    ↓ processed/nems_24h_load_scale.csv
  24小时固定模板
    ↓ GridCoupledEnv._load_grid_scenario()
  只把grid_load_scale送入OPF；绝对demand_mw不进入环境
```

### D.2 24小时聚合需求与倍率

| hour | 原始period对 | 聚合需求 MW | load scale | 聚合USEP SGD/MWh |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1–2 | 6202.3090 | 0.910053 | 160.840 |
| 1 | 3–4 | 6176.6265 | 0.906285 | 166.900 |
| 2 | 5–6 | 6049.5735 | 0.887642 | 166.645 |
| 3 | 7–8 | 5977.0215 | 0.876997 | 164.070 |
| 4 | 9–10 | 5998.9470 | 0.880214 | 165.515 |
| 5 | 11–12 | 6214.5355 | 0.911847 | 167.365 |
| 6 | 13–14 | 6605.0185 | 0.969142 | 174.040 |
| 7 | 15–16 | 7000.0015 | 1.027097 | 177.515 |
| 8 | 17–18 | 7201.3535 | 1.056641 | 177.495 |
| 9 | 19–20 | 7198.5905 | 1.056236 | 173.000 |
| 10 | 21–22 | 7099.3415 | 1.041673 | 171.145 |
| 11 | 23–24 | 6709.1200 | 0.984416 | 158.040 |
| 12 | 25–26 | 6572.3675 | 0.964351 | 157.980 |
| 13 | 27–28 | 6597.8415 | 0.968089 | 158.105 |
| 14 | 29–30 | 6796.9500 | 0.997304 | 160.365 |
| 15 | 31–32 | 6839.4830 | 1.003544 | 161.160 |
| 16 | 33–34 | 6996.9450 | 1.026648 | 161.585 |
| 17 | 35–36 | 7286.3380 | 1.069111 | 169.105 |
| 18 | 37–38 | 7546.1450 | 1.107232 | 181.655 |
| 19 | 39–40 | 7588.8420 | 1.113496 | 181.735 |
| 20 | 41–42 | 7556.9075 | 1.108811 | 179.210 |
| 21 | 43–44 | 7391.2175 | 1.084499 | 176.560 |
| 22 | 45–46 | 7095.3385 | 1.041086 | 169.320 |
| 23 | 47–48 | 6867.0345 | 1.007587 | 171.260 |

统计：需求最小5977.0215 MW、最大7588.842 MW、均值6815.327020833333 MW；倍率最小0.876997、最大1.113496、均值1.0。

该处理保留了2026-05-07相对该日均值的峰谷比例。processed文件也保留`demand_mw`，所以绝对需求可以追溯；但是`GridCoupledEnv`只读取`grid_load_scale`与`usep_sgd_per_mwh`，不把6000–7600 MW直接施加到IEEE14。当前episode始终重复这一条固定24小时模板。

### D.3 时间、重复和泄漏判断

- 当前原始文件14日均有完整48个period，日期-period无重复且按顺序排列；当前processed 24个hour无重复且单调。
- `typical`选日使用14个完整日的日均需求统计。它不是人工拼接日，而是一个真实文件中的连续完整日经聚合得到。
- 但训练与评估目前都使用同一条固定模板，没有日期级train/eval划分。因此不能宣称跨日期泛化；若未来以同一14日同时选场景和评估，会产生场景选择/评估复用风险。
- 处理脚本当前以集合检查period 1..48，并用字典按period取值；它没有显式拒绝重复period。当前文件无重复，因此当前结果不受影响。
- `hour 0`由period 1和2平均得到。仓库未记录period 1的精确时间戳及时区；只有在采用常见“period 1=00:00–00:30、period 2=00:30–01:00”的约定时，hour 0才可解释为当地00:00–01:00。源码没有UTC转换。

## E. 时间与单位检查

### E.1 同一步时间对齐

`IDCPriceEnv20D.step()`先令`t=current_step`，随后读取`T_amb[t]`、`price_t[t]`、`carbon_factor_t[t]`、`pv_t[t]`和`lambda_t[t]`，激活`arrival_time=t`的任务，用同一步功率计算成本和碳排放。`GridCoupledEnv.step()`再读取底层info中的`hour=t`，以`grid_load_scale_t[t]`和同一步`P_bus_net_kW/1000`求OPF/MEF。因此代码索引内部一致。

具体`step=10`：

```text
current_step/hour = 10
price_t[10]       = 1.05 元/kWh
carbon[10]        = 0.45 kgCO2/kWh
PV[10]            = 433.0127018922193 kW
lambda_t[10]      = 0
grid_scale[10]    = 1.0416729055404799
NEMS periods      = 21–22 的均值
固定动作P_bus_net = 606.2828973382599 kW
OPF IDC注入       = 0.6062828973382599 MW
```

- Grid使用step前的当前hour，而不是自增后的hour。
- reward中的price/carbon与同一步购电功率对应。
- 第23步使用索引23后令`terminated=True`并返回零终止观测；没有访问索引24，也没有在物理transition中重复hour 0。HARL VecEnv在终止后自动reset下一episode，但原始终止信息保留。
- `GridCoupledEnv.reset()`会用hour 0、IDC负荷0做一次初始OPF/MEF以构造reset观测；这不是episode transition，不产生reward。

### E.2 物理时钟含义

| 曲线 | hour 0的代码含义 | 时区/日期状态 | 对齐结论 |
| --- | --- | --- | --- |
| 人工IDC price | 人工0:00–1:00谷段 | 无日期、无时区 | 索引正确，现实时间不可验证 |
| 人工carbon | 夜间设定0.70 | 无日期、无时区 | 索引正确，非实测 |
| 人工temperature | 正弦索引0 | 无日期、无时区 | 索引正确，非实测 |
| 人工PV | hour0为0，hour12峰值500 kW | 无日期、无时区 | 索引正确，非实测 |
| task | 初始积压在0；随机任务到达1–18 | 无真实时钟 | 与价格没有显式相关生成 |
| NEMS scale | period 1–2平均 | 文件有日期，无时区/时间戳 | 索引正确；本地时区解释未被元数据证明 |
| BESS控制 | action_t作用于hour t | 跟随环境步 | 正确 |
| OPF/MEF | scale_t与P_bus_net_t | 跟随底层info hour | 正确 |

PV峰值在hour 12，温度峰值在hour 14，选中日NEMS需求峰值在hour 19，形状在工程上合理；但三者没有共同日期或天气来源，不能宣称真实时间相关。任务到达由独立随机数生成，也没有为价格峰谷做显式匹配。

### E.3 单位表

| 变量 | 代码单位 | 文件单位 | 换算 | 最终判断 |
| --- | --- | --- | --- | --- |
| 当前IDC price | 元/kWh | 无CSV | 无 | 与`kWh×元/kWh`一致；非真实 |
| NEMS USEP候选 | 未进入IDC成本 | SGD/MWh | 正式接入需除以1000为SGD/kWh | 当前不混用；直接接入有1000倍风险 |
| server原始功率 | W（每组数组已乘group size） | 无 | `P_IDC_t/1000`转kW | 正确 |
| IDC/PV/BESS/Grid侧功率 | kW | PV人工kW | OPF前`P_bus_net_kW/1000`转MW | 正确 |
| IEEE14/NEMS demand | MW | MW | NEMS先变无量纲倍率；不直接注入 | 正确且避免量级不兼容 |
| grid load scale | 无量纲 | 无量纲 | 无 | 正确 |
| energy | kWh | 无 | `kW×delta_t_hours(1h)` | 正确 |
| Grid reward energy | MWh | 无 | `grid_energy_kWh/1000` | 正确 |
| carbon factor | kgCO2/kWh | 无CSV | 无 | `kWh×kg/kWh=kg`正确；非真实 |
| emissions | kgCO2 | 无 | 累加 | 正确 |
| MEF | kg/MWh | 无；由工程排放因子计算 | Grid reward以MWh相乘 | 正确；排放因子非真实 |
| temperature | °C | 无CSV | 无 | 正确；非真实 |
| LMP | pandapower case14目标函数的边际成本单位；`poly_cost`列标为EUR/MW及EUR/MW² | 非NEMS | Grid reward以`MWh×LMP`计算 | 数值来自IEEE14成本曲线，不是SGD市场价；Grid reward当前关闭 |

关键归一化参考：`price_ref=1.50`；server-group生效后的`cost_ref=6000`、`carbon_ref=1500`、`lambda_ref=200000`、`queue_ref=600000`；Grid观测`grid_lmp_ref=100`、`grid_mef_ref=1000`。如果替换真实价格或碳因子，必须同步复核这些量级，不能只换CSV路径。

## F. 数据质量结果

### F.1 文件摘要和hash

| 文件 | 行数/大小 | SHA-256 | 运行角色 |
| --- | ---: | --- | --- |
| `raw/USEP_May-2026.csv` | 672行，68,567 bytes | `A6047FAB3D38A700E8BAC9E81D254BEA615FB71C8E4A48918D7961B51B8BB0D4` | 上游原始副本；运行时不读 |
| `processed/nems_24h_load_scale.csv` | 24行，4,518 bytes | `1C3319EADC5D93E9F339EEFBE3824E0E655C1AD2BEDB1C43D2BA08A16DC2296E` | 正式环境实际读取 |
| `processed/nems_all_days_hourly_profiles.csv` | 336行，59,372 bytes | `BDC671E35A8B23F615773E6DBFE754894A9012460FC7E5F0AFEECADB755D36F4` | 候选多日期数据；当前不读 |
| `processed/nems_24h_load_scale_meta.json` | 1,445 bytes | `40860E8ABB6FC5159F7FA1546EF715979E7DF6BDD364761BFACF715B46116F9F` | 处理元数据；当前不读 |

### F.2 原始CSV

- 日期范围：2026-05-01至2026-05-14；14日，每日48行，总计672行。
- 日期-period组合无重复，文件顺序单调；没有空单元格。
- `DEMAND (MW)`：全部672个数值，5779.25–7680.176，均值6749.573205；无0、负值、NaN或inf。
- `USEP ($/MWh)`：全部672个数值，103.29–349.95，均值174.085149；无0、负值、NaN或inf。
- `SOLAR(MW)`：0–1340.61，322个0，无负值。
- `LCP`和`TCL`在该文件中全部为0；`RUSEP`与USEP数值统计一致。
- `MAP`与`MAPT`各有一处`-`（CSV第567行）；当前处理脚本不读取这两个字段，所以不影响processed文件。若未来使用这些候选价格字段，必须显式处理该非数值。
- 没有显式timestamp、UTC offset或timezone列，无法验证30分钟间隔的绝对时刻；一致性只能由period 1–48推断。

### F.3 processed CSV

- 24行、单一日期2026-05-07、hour 0–23完整且唯一、行序单调。
- 所有关键字段`demand_mw/usep_sgd_per_mwh/solar_mw/grid_load_scale`均可解析且有限；无空值、负值、重复hour。
- load scale均值为1.0000000000000002（浮点误差范围），符合按单日均值归一化。
- `source_file`列和meta内的路径是生成机器的绝对路径。环境不读取该列，所以代码运行可移植；但数据溯源记录在仓库移动后会显得机器相关。
- processed数据没有记录时区、外部来源URL、许可证、原始下载hash或处理脚本commit。当前hash是本次审计现算，不是数据manifest的一部分。

### F.4 真实性边界

仓库内容足以证明“这些数值来自仓库内名为NEMS/USEP的CSV，并按可审计脚本处理”，不足以证明“该原始CSV确由官方渠道下载且未经改变”。正式论文需要本地manifest补齐来源URL/发布机构/下载日期/许可证/原始hash/时区；本次遵守限制，没有联网核验。

## G. 场景配置表

### G.1 正式experiment cases

| case | 数据配置 | reward覆盖 | Grid覆盖 | 正式入口可选择性 |
| --- | --- | --- | --- | --- |
| `main` / alias `report_main` | 基础`DATA_CONFIG`，即IDC外部路径均None；NEMS Grid全局配置不变 | 无 | 无 | YAML中当前为`main`；无CLI覆盖 |
| `no_bess` | 同main | 无 | 无 | 可通过另一份/修改后的YAML传入；无CLI覆盖 |
| `carbon_w0` | 同main | `reward_carbon_weight=0.0` | 无 | 同上 |
| `carbon_w03` | 同main | `reward_carbon_weight=0.30`，与当前main相同 | 无 | 同上 |
| `carbon_w05` | 同main | `reward_carbon_weight=0.50` | 无 | 同上 |

- `main`只表示基础环境/奖励/数据配置，不表示“真实数据”。现有case都使用同一DATA_CONFIG和同一NEMS Grid全局配置；没有真实数据case。
- `train/train_harl_mappo_short.py`仅定义`--scenario`，没有`--experiment-case`。正式入口会读取YAML里的`env.experiment_case`，所以技术上可用另一配置文件选择case，但不能在当前CLI直接覆盖。
- `configs/single_idc_scenarios.py`含small baseline、normal、strong、stress等合成Safe PPO场景；`configs/multi_idc_scenarios.py`含多IDC诊断场景。它们不是`experiment_cases.py`中的case，也没有进入正式双Agent MAPPO工厂。
- 未发现名为sin的正式experiment case。未发现现有experiment case引用失效数据文件；这些case自身不含文件路径。
- `get_experiment_case()`使用`deepcopy`，不存在共享嵌套配置被跨运行修改的问题。

### G.2 一个正式episode的固定与随机组成

```text
固定24小时人工price
+ 固定24小时人工carbon
+ 固定24小时人工temperature
+ 固定24小时人工PV
+ 固定24小时NEMS需求派生grid scale
+ 固定IEEE14网络和人工发电机排放因子
+ 每个环境实例构造时seed随机生成一次server参数
+ 每次reset继续使用task RNG生成一组新task
+ 训练时Actor随机采样动作
```

- 同一环境实例中server参数episode间不变；task会随reset重新抽样，但给定seed时整个episode序列可复现。
- price/carbon/temperature/PV/Grid scale每episode完全重复，策略存在记忆固定24小时模板的风险。
- 固定Grid scale和不清空的per-worker缓存会增加跨episode复用机会，但IDC净负荷仍受task和动作影响；未运行训练，不能据此断言实际命中率一定“极高”。
- 正式评估应至少加入未见日期/曲线；当前单日模板只能测同场景复现，不能测跨日泛化。

## H. 正式环境实测

### H.1 方法

- 通过`marl.envs.harl_env_factory.make_harl_train_env(seed=2026, n_rollout_threads=1, scenario="idc_bess_padding", experiment_case="main")`构造正式环境。
- 调用一次reset；观测shape为`(1,2,288)`，centralized state shape为`(1,2,294)`。
- 固定动作是shape `(1,2,22)`的全0.5数组。IDC 22维均为0.5，BESS有效第0维为0.5，对应底层BESS中性动作；未创建Actor、未执行policy update。

### H.2 reset后的实际24小时数组

```text
price =
[0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.65, 0.65, 0.65,
 1.05, 1.05, 1.05, 1.05, 1.05, 0.65, 0.65, 0.65, 1.05, 1.05,
 1.05, 0.65, 0.65, 0.65]

carbon =
[0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.60, 0.60, 0.60,
 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.60, 0.60, 0.80, 0.80,
 0.80, 0.80, 0.60, 0.60]

temperature_C =
[20.669873, 20.170371, 20.000000, 20.170371, 20.669873, 21.464466,
 22.500000, 23.705905, 25.000000, 26.294095, 27.500000, 28.535534,
 29.330127, 29.829629, 30.000000, 29.829629, 29.330127, 28.535534,
 27.500000, 26.294095, 25.000000, 23.705905, 22.500000, 21.464466]

pv_kW =
[0, 0, 0, 0, 0, 0, 0, 129.409523, 250.000000, 353.553391,
 433.012702, 482.962913, 500.000000, 482.962913, 433.012702,
 353.553391, 250.000000, 129.409523, ~0, 0, 0, 0, 0, 0]

grid_load_scale =
[0.910053, 0.906285, 0.887642, 0.876997, 0.880214, 0.911847,
 0.969142, 1.027097, 1.056641, 1.056236, 1.041673, 0.984416,
 0.964351, 0.968089, 0.997304, 1.003544, 1.026648, 1.069111,
 1.107232, 1.113496, 1.108811, 1.084499, 1.041086, 1.007587]

task_arrival_workload =
[30000.000000, 11721.913573, 12833.832532, 25440.783122,
 155367.883628, 62869.885917, 59675.190337, 43766.199701,
 30617.901913, 6703.642614, 0, 90089.470876, 85005.294125,
 34100.130961, 43658.135710, 0, 9690.727184, 11250.513694,
 75274.446059, 0, 0, 0, 0, 0]
```

运行时数组与代码和processed CSV逐项一致。WT为24个0。

### H.3 初始server/BESS/Grid摘要

| 项 | 实测值 |
| --- | ---: |
| server groups | 20 |
| group size | 100 |
| effective server count | 2000 |
| group `P_idle`范围/均值 | 17,418.826–23,735.698 W；均值20,297.411 W |
| group `P_max`范围/均值 | 48,308.172–70,723.675 W；均值59,960.718 W |
| `C_server`范围/均值 | 1,877.253–2,867.539；均值2,474.090 |
| 总idle功率 | 405.948213 kW |
| 总max功率 | 1,199.214358 kW |
| 总group capacity | 49,481.790234 |
| BESS容量/充放功率 | 10,000 kWh；2,000/2,000 kW |
| BESS初始SOC | 0.5 |
| Grid | IEEE14，base 100 MVA，AC OPF，IDC在IEEE bus 9（pandapower index 8），MEF开启 |
| Grid reward | 关闭 |

### H.4 固定动作24步结果

| h | price | carbon | PV kW | scale | IDC kW | P_bus_net kW | LMP | MEF+ kg/MWh | reward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.35 | 0.70 | 0.000 | 0.910053 | 1009.448 | 1009.448 | 39.8936 | 307.8616 | -0.073435 |
| 1 | 0.35 | 0.70 | 0.000 | 0.906285 | 1007.519 | 1007.519 | 39.8681 | 312.0693 | -0.074800 |
| 2 | 0.35 | 0.70 | 0.000 | 0.887642 | 1006.868 | 1006.868 | 39.7388 | 316.7895 | 0.048435 |
| 3 | 0.35 | 0.70 | 0.000 | 0.876997 | 1007.519 | 1007.519 | 39.6648 | 316.9081 | -0.033148 |
| 4 | 0.35 | 0.70 | 0.000 | 0.880214 | 1009.448 | 1009.448 | 39.6872 | 320.5086 | -0.204015 |
| 5 | 0.35 | 0.70 | 0.000 | 0.911847 | 1012.581 | 1012.581 | 39.9056 | 304.3630 | -0.262768 |
| 6 | 0.35 | 0.70 | 0.000 | 0.969142 | 1016.788 | 1016.788 | 40.0968 | 168.4429 | -0.390569 |
| 7 | 0.65 | 0.60 | 129.410 | 1.027097 | 1021.871 | 892.461 | 40.2225 | 168.2460 | -0.358646 |
| 8 | 0.65 | 0.60 | 250.000 | 1.056641 | 1027.559 | 777.559 | 40.2852 | 166.1033 | -0.295291 |
| 9 | 0.65 | 0.60 | 353.553 | 1.056236 | 1033.506 | 679.953 | 40.2828 | 166.2003 | -0.341797 |
| 10 | 1.05 | 0.45 | 433.013 | 1.041673 | 1039.296 | 606.283 | 40.2502 | 167.7635 | -0.318103 |
| 11 | 1.05 | 0.45 | 482.963 | 0.984416 | 1044.471 | 561.508 | 40.1235 | 168.4357 | -0.409928 |
| 12 | 1.05 | 0.45 | 500.000 | 0.964351 | 1048.577 | 548.577 | 40.0791 | 168.4978 | -0.614674 |
| 13 | 1.05 | 0.45 | 482.963 | 0.968089 | 1051.220 | 568.257 | 40.0877 | 168.4612 | -0.683549 |
| 14 | 1.05 | 0.45 | 433.013 | 0.997304 | 1052.133 | 619.120 | 40.1528 | 168.4281 | -0.792613 |
| 15 | 0.65 | 0.45 | 353.553 | 1.003544 | 1051.220 | 697.667 | 40.1677 | 168.4179 | -0.871444 |
| 16 | 0.65 | 0.60 | 250.000 | 1.026648 | 1048.577 | 798.577 | 40.2201 | 168.2852 | -0.732260 |
| 17 | 0.65 | 0.60 | 129.410 | 1.069111 | 1044.471 | 915.061 | 40.3148 | 164.0168 | -0.925908 |
| 18 | 1.05 | 0.80 | ~0 | 1.107232 | 1039.296 | 1039.296 | 40.3880 | 154.7698 | -1.186635 |
| 19 | 1.05 | 0.80 | 0.000 | 1.113496 | 1033.506 | 1033.506 | 40.3993 | 154.3705 | -1.019349 |
| 20 | 1.05 | 0.80 | 0.000 | 1.108811 | 1027.559 | 1027.559 | 40.3907 | 154.5632 | -1.181446 |
| 21 | 0.65 | 0.80 | 0.000 | 1.084499 | 1021.871 | 1021.871 | 40.3455 | 158.2508 | -1.253409 |
| 22 | 0.65 | 0.60 | 0.000 | 1.041086 | 1016.788 | 1016.788 | 40.2551 | 167.7601 | -1.106708 |
| 23 | 0.65 | 0.60 | 0.000 | 1.007587 | 1012.581 | 1012.581 | 40.1813 | 168.4060 | -3.317177 |

验证汇总：24/24 OPF成功；24/24 MEF成功；所有列有限；hour序列严格为0–23；第23小时`terminated=True`，无truncated。MEF-也全部有限，范围154.3767–319.8943 kg/MWh。

## I. fallback与重点风险

### I.1 失败行为

| 数据 | 加载失败后的行为 | 是否报警 | 是否阻止正式训练 |
| --- | --- | --- | --- |
| IDC price CSV | path为None时有意返回None并使用人工曲线；若配置了不存在路径、错列、坏值、非24长度或NaN/inf则抛异常 | 是，异常 | 是 |
| carbon CSV | 同上；None使用人工碳曲线 | 是，异常 | 是 |
| temperature CSV | 同上；None使用人工温度曲线 | 是，异常 | 是 |
| PV CSV | None且`use_default_pv_curve=True`时有意生成正弦PV；配置路径错误/错列/坏值/负值则抛异常 | 是，异常 | 是 |
| WT/其他optional series | None时有意返回全0；配置路径错误或坏值则抛异常 | 是，异常 | 是 |
| task数据 | 没有CSV路径，始终随机合成；生成异常从reset向上传播 | 是，异常 | 是 |
| NEMS Grid CSV | `_load_grid_scenario()`捕获异常并用常数倍率1.0继续构造 | 只写`grid_scenario_message/source=fallback`，不立即抛出 | **正式入口会被`validate_grid_startup()`拒绝**；非正式直接构造可继续 |

因此，不存在“已配置IDC CSV但读取失败后静默改用人工曲线”的路径；`None`是显式选择默认模式。真正存在静默降级的是Grid wrapper内部，但第二部分增加的正式启动fail-fast仍有效：`train/train_harl_mappo_short.py::prepare_training_environment()`用一次性probe reset调用`validate_grid_startup()`，拒绝`source`以`fallback`开头、Grid未启用、非AC、MEF未启用或初始OPF/MEF失败。

### I.2 风险清单

1. **命名风险**：把当前main描述成真实数据场景会掩盖绝大多数IDC输入是合成的事实。
2. **价格混淆风险**：NEMS demand派生倍率真实进入Grid；NEMS USEP没有进入IDC成本。两者不能互换。
3. **1000倍风险**：未来把SGD/MWh直接乘kWh会放大成本1000倍。
4. **币种风险**：IEEE14 LMP成本曲线列标EUR，NEMS USEP为SGD，人工IDC价格标元；当前三者未相加，但未来Grid reward/真实价格实验必须明确币种。
5. **时间语义风险**：原始CSV没有时区和timestamp；period到当地钟点的解释未由manifest证明。
6. **跨源相关性风险**：NEMS固定在2026-05-07，其他曲线无日期，不存在可证明的天气、碳、价格、负荷联合相关。
7. **固定模板风险**：每episode重复同一price/carbon/temp/PV/grid scale，策略可能记忆hour而非学习跨日稳健调度。
8. **loader错位边缘风险**：`GridCoupledEnv._load_grid_scenario()`先过滤非法scale，再取前24个，但USEP直接取原rows前24个。若未来文件中某个scale非法而仍剩至少24个有效scale，scale可能前移且与USEP错位。当前24行全部有效，不影响本次短训练。
9. **绝对负荷语义风险**：环境使用的是IEEE14基准负荷乘NEMS相对倍率，不是新加坡系统6000–7600 MW的电网复现。
10. **数据来源风险**：缺少上游URL、许可证、时区、下载日期和原始官方hash。
11. **本机路径风险**：processed CSV和meta内部保存旧机器绝对路径；运行路径本身由项目根解析，不受影响，但追溯文本不可移植。
12. **归一化风险**：真实价格/碳因子接入后继续沿用人工参考值会压缩或放大观测/reward。

## J. metadata充分性

| 项目 | 当前是否记录 | 证据/缺口 |
| --- | --- | --- |
| 实际Grid文件路径 | 部分 | `run_metadata.json`和`resolved_config.json.runtime`记录运行时绝对`grid_scenario_source` |
| IDC DATA_CONFIG | 否 | resolved HARL配置只含`env.experiment_case`，没有展开`DATA_CONFIG` |
| GRID_SCENARIO_CONFIG | 否 | 只记录source/message，未保存完整配置 |
| CSV hash | 否 | 本报告计算了hash，但正式run metadata不计算 |
| 数据日期 | 否 | metadata不读取processed的2026-05-07 |
| 原始时间分辨率/小时分辨率 | 否 | 未记录30min→1h |
| 聚合与归一化方式 | 否 | 未记录mean/typical/day-mean |
| 单位 | 否 | 未记录SGD/MWh、MW、无量纲等 |
| 真实/合成标记 | 否 | 没有逐曲线provenance |
| processed上游来源 | 否 | run metadata不读取meta JSON；processed自身只有机器绝对source_file |
| 上游外部URL/许可证/时区 | 否 | 仓库数据元信息本身也缺少 |

因此当前run可以定位它打开了哪个processed路径，但不能仅凭输出目录重建“文件内容、选中日期、处理方式、单位和各IDC曲线真实性”。这不阻塞工程短训练，却阻塞论文级可追溯性。

## K. 问题分类

### 阻塞MAPPO短训练

- **未发现。** 在明确把本轮称为“混合/合成工程短训练”而非真实数据实验的前提下，当前实际文件存在、24行有效、单位换算链正确、hour索引一致，正式启动fail-fast有效，固定动作episode通过。

### 必须在正式长训练前修正

- 明确场景命名和实验说明：不能把`main`等同于真实数据场景。
- 为NEMS数据补齐来源、日期、时区、单位、聚合、许可证和hash manifest。
- 在run metadata保存实际文件hash、日期、处理方法、完整`DATA_CONFIG/GRID_SCENARIO_CONFIG`及逐曲线真实/合成标签。
- 建立多日期train/eval场景，避免所有episode和评估共享一条固定24小时曲线。
- 如果论文目标需要真实购电成本，应接入经`/1000`换算的真实价格，并重新标定`price_ref/cost_ref`；不能只把USEP列直接塞进当前公式。

### 建议改进

- 增加真实或有依据的碳因子、温度、PV和IDC workload数据，并按共同日期/时区对齐。
- 让处理脚本显式拒绝重复period、输出timezone和相对source路径。
- 区分NEMS USEP、IEEE14 LMP和IDC购电合同价，明确币种与是否含网络/零售费用。
- 评估增加未见日期、峰值日、低负荷日和极端价格日，并固定场景清单。
- 对Grid loader逐行联动校验hour/scale/USEP，避免未来过滤后错位。

### 可以保持现状

- IEEE14作为工程集成与算法基线的标准测试系统。
- NEMS单日需求派生倍率用于MAPPO短训练的动态Grid形状。
- 当前人工price/carbon/temp/PV用于可控、可复现的短训练基线。
- 当前任务和server的seed复现机制。
- 当前kW/kWh、MW/MWh、MEF单位转换以及正式Grid启动fail-fast。

## L. 第五部分判断

**选择2：基本通过，短训练可用，但正式实验需补充真实数据。**

理由：当前场景并非纯合成——Grid负荷形状确实由仓库内NEMS格式需求数据聚合归一化得到；同时它也绝不是完整真实数据场景，因为IDC价格、碳、温度、PV、任务、server、BESS和网络主体均为人工、随机或标准测试系统。当前数据链、单位和step对齐足够支持MAPPO工程短训练，但不足以支持真实数据性能结论。

## M. 最小修正建议

本节只给建议，未修改任何下列文件。

### M.1 短训练前必须修正

- **代码/数据层无必须修正项。** 首次MAPPO短训练只需在运行说明和输出命名中明确“hybrid engineering baseline / NEMS-derived grid load scale”，禁止标注为真实数据实验。
- 启动前继续保留现有`validate_grid_startup()`；不要绕过Grid CSV fallback检查。

### M.2 MAPPO基线后再修正

```text
建议新增：
- data/grid_scenarios/nems_singapore/data_manifest.json
  职责：来源URL/机构、下载日期、许可证、时区、原始与processed hash、单位、日期范围、聚合/归一化方法。

建议新增或扩展：
- configs/experiment_cases.py
  职责：显式区分hybrid_baseline、真实价格候选和多日期场景；case名称表达实际数据内容。
- train/train_harl_mappo_short.py
  职责：把完整DATA_CONFIG、GRID_SCENARIO_CONFIG、数据日期/hash/单位/provenance写入resolved config和metadata。
- scripts/build_grid_load_scale_from_nems.py
  职责：拒绝重复period，记录timezone，输出相对上游路径和处理脚本版本/hash。
- env_wrappers/grid_coupled_env.py
  职责：未来按同一行同时校验hour、scale和USEP，禁止过滤导致列错位。
```

### M.3 正式论文实验前必须修正

```text
建议新增：
- 独立的真实IDC价格预处理产物与manifest
  职责：明确SGD/MWh→SGD/kWh、币种、日期、时区、缺失/尖峰策略，不在环境中隐式clip。
- 多日期train/eval场景清单
  职责：日期级不重叠、固定评估日期、峰值/低负荷/高价场景覆盖。
- 真实碳因子、温度、PV和workload数据接口/manifest
  职责：共同时间轴、单位和来源可追溯。

需要复核配置：
- configs/config_ultimate.py
  职责：真实数据路径、price_ref、cost_ref、carbon_ref及Grid归一化参考值。
```

### M.4 建议保持不动

```text
- marl/bridges/**：Bridge与padding不属于本问题。
- marl/runners/**：MAPPO runner不属于数据接入问题。
- HARL有界Box动作实现：与真实数据来源无关。
- grid_model/opf_solver.py、grid_model/mef_calculator.py：当前单位链和短训练验证正常。
- envs/idc_price_env.py的现有物理/reward逻辑：在正式数据方案确认前不要临时改公式或clip。
```

## 审计边界

- 本次未联网、未下载数据、未安装依赖。
- 未修改数据、环境配置、reward、训练、Bridge、runner、OPF/MEF或cache。
- 未运行MAPPO Actor、5-update或40-update训练，未执行policy update。
- 只执行了正式环境工厂的reset与固定动作24步轻量验证。
- 除本Markdown报告外未生成新CSV或其他产物；未进入第六部分。
