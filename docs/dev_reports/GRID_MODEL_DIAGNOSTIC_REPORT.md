# Grid Model Diagnostic Report

## 1. Modified Files

- `grid_model/grid_case.py`
- `grid_model/ieee14_loader.py`
- `grid_model/opf_solver.py`
- `grid_model/emission_model.py`
- `grid_model/mef_calculator.py`
- `grid_model/__init__.py`
- `scripts/smoke_test_grid_model.py`
- `.gitignore`
- `GRID_MODEL_DIAGNOSTIC_REPORT.md`

## 2. Generator Voltage Warning

Pandapower `case14()` can contain generator voltage setpoints that are outside the corresponding bus OPF voltage limits. During OPF, pandapower warns:

```text
gen vm_pu > bus max_vm_pu for gens [2 3]. Setting bus limit for these gens.
```

The loader now calls `harmonize_generator_voltage_limits(net)` immediately after `pandapower.networks.case14()`.

The function checks each generator's `vm_pu` and its connected bus:

- If `gen.vm_pu > bus.max_vm_pu`, it raises the bus `max_vm_pu` to the generator setpoint.
- If `gen.vm_pu < bus.min_vm_pu`, it lowers the bus `min_vm_pu` to the generator setpoint.

This does not modify IEEE14 topology, branches, loads, or generator placement. It only makes pandapower's OPF feasibility bounds consistent with generator voltage setpoints.

## 3. Bus ID Semantics

The code now distinguishes:

- IEEE bus number: the original test-system number, typically `1..14`.
- Pandapower bus index: the internal pandapower index, typically `0..13`.

`GridCase` now stores:

- `ieee_bus_number_to_bus_index`
- `bus_index_to_ieee_bus_number`

Helper functions:

- `get_bus_index_by_ieee_number(grid_case, ieee_bus_number)`
- `get_ieee_number_by_bus_index(grid_case, bus_index)`

The smoke test now uses:

```python
idc_ieee_bus_number = 9
idc_bus_idx = get_bus_index_by_ieee_number(grid_case, idc_ieee_bus_number)
```

For standard pandapower IEEE14, this maps:

```text
IDC IEEE bus number = 9
IDC pandapower bus index = 8
```

All OPF and MEF calls use the pandapower bus index.

## 4. Generator Emission Diagnostics

`emission_model.py` now includes `build_generator_emission_table()`.

For each OPF result it reports:

- `gen_id`
- `gen_power_mw`
- `emission_factor_kg_per_mwh`
- `gen_emission_kg`

The smoke test prints this table after DC OPF and AC OPF, plus highlights for:

- largest generator output
- highest emission factor
- largest emission contribution

The emission formula remains unchanged.

## 5. MEF Base/Plus/Minus Diagnostics

`MEFResult` now records:

- `base_gen_power_mw`
- `plus_gen_power_mw`
- `minus_gen_power_mw`
- `delta_gen_power_plus_mw`
- `delta_gen_power_minus_mw`
- base/plus/minus total generation
- base/plus/minus network loss

The smoke test prints:

```text
gen_id | base_mw | plus_mw | minus_mw | delta_plus_mw | delta_minus_mw | EF
```

This shows which generator increases output when load is added and which generator reduces output when load is removed.

## 6. DC-MEF and AC-MEF Difference Diagnostics

The smoke test now prints a `[MEF DIAGNOSTIC]` block with:

- DC-MEF plus/minus
- AC-MEF plus/minus
- DC and AC base total emissions
- DC and AC base total generation
- DC and AC base network loss
- main plus/minus marginal generators under DC and AC

The diagnostic output explicitly notes:

- `DC/AC marginal generator differs` when the dominant marginal generator is different.
- `AC OPF includes network loss, which may change dispatch and MEF` when AC loss is non-trivial.
- `MEF is sensitive to generator emission factors`.

No attempt was made to force DC-MEF and AC-MEF to match.

## 7. Smoke Test Result

Direct requested commands could not run in this shell because `python` is not on PATH:

```text
python : The term 'python' is not recognized...
```

The equivalent compile check passed using the Codex bundled Python runtime:

```powershell
& 'C:\Users\bulio\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile (Get-ChildItem grid_model -Filter *.py).FullName scripts\smoke_test_grid_model.py
```

The bundled runtime still does not expose `pandapower`, so its smoke run stops cleanly at dependency loading:

```text
IEEE14 smoke test skipped: pandapower is not available.
Reason: pandapower is required to load the IEEE 14-bus grid case. Install it with `pip install pandapower` in the active Python environment.
```

A local unit-style check of the voltage harmonization and IEEE/pandapower bus mapping helpers passed with a small in-memory network object.

In the user's pandapower-enabled environment, run:

```powershell
python -m scripts.smoke_test_grid_model
```

The expected output now includes the IDC IEEE bus number, pandapower bus index, OPF summaries, generator emission tables, MEF dispatch tables, and the `[MEF DIAGNOSTIC]` section.

## 8. Follow-Up Recommendations

- Calibrate generator emission factors with a documented source instead of relying on the current first-pass defaults.
- Test several IDC access buses, especially electrically weak or congested buses.
- Sweep multiple `delta_p_mw` values to check MEF numerical stability.
- Keep GridCoupledEnv integration as the next separate step under `env_wrappers/`, without embedding grid physics in the IDC environment.
