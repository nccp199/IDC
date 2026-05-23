# Grid Coupled Env Report

## 1. Files

Added:

- `env_wrappers/grid_coupled_env.py`
- `scripts/smoke_test_grid_coupled_env.py`
- `docs/dev_reports/GRID_COUPLED_ENV_REPORT.md`

Modified:

- `configs/config_ultimate.py`
- `env_wrappers/__init__.py`
- `grid_model/mef_calculator.py`

## 2. Design

`GridCoupledEnv` is a Gymnasium wrapper around the existing IDC environment. The base IDC environment still owns task generation, task scheduling, BESS behavior, PUE/COP, cost, carbon, and reward calculation. The wrapper reads the base environment's step `info`, maps `P_grid_kW` into IEEE14 as an added load, and appends grid feedback fields to `info`.

The observation and action spaces are unchanged in this first version.

## 3. Why A Wrapper

The grid model is intentionally kept outside `envs/idc_price_env.py`. This avoids mixing OPF/MEF physics into IDC task-scheduling logic, keeps the original environment usable for prior experiments, and makes grid coupling easy to enable, disable, or replace.

## 4. Power Conversion

The wrapper reads:

```python
P_grid_kW = info["P_grid_kW"]
idc_load_mw = max(P_grid_kW / 1000.0, 0.0)
```

The first version treats IDC grid purchase as a non-negative additional load at the configured IEEE bus.

## 5. OPF / MEF Flow

Each step:

1. Call `base_env.step(action)`.
2. Convert `P_grid_kW` to `idc_load_mw`.
3. Run `solve_opf()` with the configured OPF mode and IDC bus.
4. Compute OPF emissions from generator dispatch.
5. If enabled, run `calculate_nodal_mef()` around the current IDC operating point.
6. Extract `GridMetricResult`.
7. Append grid fields to `info`.
8. Keep reward unchanged unless `enable_grid_reward=True`.

## 6. Added Info Fields

The wrapper appends:

- Grid identity and status: `grid_enabled`, `grid_case_name`, `grid_opf_mode`, `grid_opf_success`, `grid_opf_message`, `grid_idc_ieee_bus_number`, `grid_idc_bus_index`, `grid_idc_load_mw`
- LMP / MEF: `grid_lmp`, `grid_mef_plus`, `grid_mef_minus`, `grid_mef_success`, `grid_mef_message`
- System OPF: `grid_total_generation_cost`, `grid_total_emission_kg`, `grid_total_load_mw`, `grid_total_generation_mw`, `grid_network_loss_mw`
- Grid safety: `grid_min_voltage_pu`, `grid_max_voltage_pu`, `grid_max_line_loading_percent`, `grid_voltage_violation_count`, `grid_line_overload_count`, `grid_security_penalty`
- SAFE RL placeholders: `safe_cost_voltage`, `safe_cost_line`, `safe_cost_opf`, `safe_cost_total`
- Reward diagnostics: `base_reward`, `grid_reward_penalty`, `grid_adjusted_reward`

## 7. Reward Behavior

`enable_grid_reward` defaults to `False`, and all grid reward weights default to zero. Therefore `grid_adjusted_reward == base_reward` in the first version. The penalty code is present for later experiments but inactive by default.

## 8. Smoke Test

Smoke test command:

```powershell
python -m scripts.smoke_test_grid_coupled_env
```

In the current Codex shell, `python` is not on PATH, so the equivalent bundled Python runtime was used:

```powershell
& 'C:\Users\bulio\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m scripts.smoke_test_grid_coupled_env
```

Result:

- Episode length: 24 hours
- OPF success: 24 / 24
- MEF success: 24 / 24
- OPF failures: 0
- MEF failures: 0
- Average grid LMP: `40.149655`
- Average MEF plus: `168.428706 kgCO2/MWh`
- Max grid line loading: `1.235215%`
- Min grid voltage: `1.015573 pu`
- `grid_adjusted_reward == base_reward` for every step because `enable_grid_reward=False`

Compatibility checks:

- `python -m scripts.smoke_test_grid_model` equivalent run passed.
- `python -m scripts.scan_ieee14_bus_sensitivity` equivalent run passed and regenerated the DC+AC node sensitivity outputs.

## 9. Follow-Up Plan

1. Add selected grid fields to observations in a second-stage wrapper.
2. Add grid metrics to evaluation scripts.
3. Train a PPO baseline with grid info disabled/enabled for ablation.
4. Add SAFE RL costs using the `safe_cost_*` fields.
5. Upgrade to MAPPO+CTDE with attention over node-level grid features.
