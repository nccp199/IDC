# Grid Model Skeleton Report

## 1. Added Files

- `grid_model/grid_case.py`
- `grid_model/ieee14_loader.py`
- `grid_model/opf_solver.py`
- `grid_model/emission_model.py`
- `grid_model/mef_calculator.py`
- `grid_model/grid_metrics.py`
- `scripts/smoke_test_grid_model.py`
- `GRID_MODEL_SKELETON_REPORT.md`

## 2. File Responsibilities

- `grid_model/grid_case.py`: shared dataclasses for `GridCase`, `OPFResult`, `MEFResult`, and `GridMetricResult`.
- `grid_model/ieee14_loader.py`: loads `pandapower.networks.case14()` and wraps it as a `GridCase`. It does not solve OPF.
- `grid_model/opf_solver.py`: provides the unified `solve_opf()` entry point and separate DC/AC OPF functions. It deep-copies the pandapower network before load scaling and IDC load adjustment.
- `grid_model/emission_model.py`: builds first-pass generator emission factors and computes total kgCO2 emissions from dispatch.
- `grid_model/mef_calculator.py`: computes nodal plus/minus MEF by running base, plus-load, and minus-load OPF cases.
- `grid_model/grid_metrics.py`: extracts voltage, line loading, infeasibility, and simple security penalty metrics from `OPFResult`.
- `scripts/smoke_test_grid_model.py`: minimal runtime smoke test for IEEE14 loading, DC OPF, AC OPF, DC-MEF, and AC-MEF.
- `grid_model/__init__.py`: exports the main grid-model interfaces.

## 3. DC OPF and AC OPF Interfaces

The main interface is:

```python
solve_opf(
    grid_case,
    mode="dc",
    idc_bus_id=None,
    idc_load_mw=0.0,
    load_scale=1.0,
)
```

- `mode="dc"` calls `solve_dc_opf()` and then `pandapower.rundcopp()`.
- `mode="ac"` calls `solve_ac_opf()` and then `pandapower.runopp()`.
- The original `GridCase.raw_network` is never mutated; the solver uses `copy.deepcopy()`.
- IDC load is modeled as a fixed additional nodal load in MW.
- Failures return `OPFResult(success=False, message=...)` instead of raising through the public interface.
- DC fields that may not exist, such as bus voltages or line loading, are returned as empty dictionaries or `nan`.

## 4. MEF Calculation Flow

`calculate_nodal_mef()` performs:

1. Run base OPF.
2. Compute base emissions from generator dispatch.
3. Add `delta_p_mw` load at the target bus and run OPF.
4. Compute plus-case emissions.
5. Add `-delta_p_mw` at the same bus and run OPF.
6. Compute minus-case emissions.
7. Compute `MEF_plus = (plus_emission - base_emission) / delta_p_mw`.
8. Compute `MEF_minus = (base_emission - minus_emission) / delta_p_mw`.

The first version assumes a one-hour interval, so the reported unit is kgCO2/MWh.

## 5. Smoke Test Result

Command attempted with the bundled Codex Python runtime because `python` is not on PATH in this shell:

```powershell
& 'C:\Users\bulio\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m scripts.smoke_test_grid_model
```

Compile check passed:

```powershell
& 'C:\Users\bulio\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile (Get-ChildItem grid_model -Filter *.py).FullName scripts\smoke_test_grid_model.py
```

The literal `grid_model/*.py` form was not expanded by this PowerShell/native-command combination, so the check used `Get-ChildItem` to pass the same file list explicitly.

Smoke test completed without crashing, but skipped OPF execution because `pandapower` is not installed:

```text
IEEE14 smoke test skipped: pandapower is not available.
Reason: pandapower is required to load the IEEE 14-bus grid case. Install it with `pip install pandapower` in the active Python environment.
```

## 6. Dependency Issues

The bundled Python runtime currently does not have `pandapower` installed:

```text
ModuleNotFoundError: No module named 'pandapower'
```

Install `pandapower` in the active Python environment to run IEEE14 loading and OPF:

```powershell
pip install pandapower
```

The loader intentionally raises a clear `ImportError` when `pandapower` is unavailable.

## 7. Future GridCoupledEnv Integration

The next integration step should keep `envs/idc_price_env.py` unchanged and add a coupling layer under `env_wrappers/`, for example `GridCoupledEnv`.

Recommended flow:

1. The IDC environment produces IDC power demand for the current step.
2. The wrapper maps IDC demand to `idc_bus_id` and `idc_load_mw`.
3. The wrapper calls `solve_opf()` or a cached/accelerated grid service.
4. The wrapper appends selected `grid_obs` fields to the policy observation.
5. The wrapper injects grid penalties or constraints into reward shaping without embedding grid physics inside the IDC environment.

## 8. Future MAPPO+CTDE Wiring

For MAPPO+CTDE with attention, keep grid information as structured centralized context:

- `grid_obs`: selected nodal LMP, voltage, line loading, total load, total generation, and IDC bus indicators.
- `grid_metrics`: `GridMetricResult` fields for centralized critic inputs and reward diagnostics.
- Actor observation: only local or allowed operational signals.
- Centralized critic observation: full IDC state plus grid metrics and nodal summaries.
- Attention keys: node-level features such as bus id, LMP, voltage, MEF, and electrical/security indicators.
