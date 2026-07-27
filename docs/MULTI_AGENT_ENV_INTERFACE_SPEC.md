# Multi-Agent Environment Interface Specification

Status: first generic two-agent interface over the current single-agent environment. It does not provide a HARL bridge or any learning algorithm.

## 1. Composition and responsibilities

The execution chain is `IDCPriceEnv20D -> GridCoupledEnv -> IDCGridMultiAgentEnv`. The outer environment never bypasses `GridCoupledEnv`; transitions use only `self.env.reset(...)` and `self.env.step(flat_action)`.

Agents are `("idc", "bess")`. The IDC agent schedules server groups and task preferences. The BESS agent controls the existing final storage action. Both receive the unchanged global scalar reward returned by `GridCoupledEnv`.

## 2. Action interface

```python
action_dict = {
    "idc": np.ndarray(shape=(22,), dtype=float32),
    "bess": np.ndarray(shape=(1,), dtype=float32),
}
```

All values must be finite and in `[0, 1]`; invalid values raise an exception and are not clipped. The IDC vector is copied to legacy `action[0:22]`: dimensions 0-19 are server-group execution intensities, 20 is urgent preference, and 21 is continuity preference. The BESS value is copied unchanged to legacy `action[22]`. Its `[0,1] -> [-1,1]` charge/discharge mapping remains exclusively inside `IDCPriceEnv20D`.

## 3. Observations

The base formula is:

```text
base_obs_dim = 6 + 10 + 6*N + 6*horizon
wrapped_obs_dim = base_obs_dim + 8
```

With `N=20` and `horizon=24`, these are 280 and 288.

IDC observation is the unmodified wrapped observation, in the order documented by `ENVIRONMENT_INTERFACE_SPEC.md`: 6 global, 10 task-pool, `6*N` server, `6*horizon` forecast, then 8 grid features. Its current dimension is 288.

BESS observation has this fixed order:

1. Current 6 global features, legacy wrapped slice `[0:6]`.
2. Forecast features, the last `6*horizon` values of the base observation; current slice `[136:280]` and dimension 144.
3. The final 8 grid features; current slice `[280:288]`.
4. Six supplemental values in order: `bess_soc`, `bess_energy_kWh`, `P_IDC_kW`, `P_grid_kW`, `bess_charge_power_kW`, `bess_discharge_power_kW`.

Thus `bess_obs_dim = 6 + 6*horizon + 8 + 6`, currently 164. It deliberately excludes the 10 task-pool details and all `6*N` server-group features.

## 4. Centralized state

State is the complete wrapped observation followed by the same six supplemental values and never includes the joint action:

```text
state_dim = wrapped_obs_dim + 6
          = (6 + 10 + 6*N + 6*horizon + 8) + 6
```

The current dimension is 294. All observations and state are finite `float32` arrays. Their declared spaces use unbounded boxes because the legacy features are not uniformly restricted to `[0,1]`.

## 5. Calls and termination

```python
obs_dict, state, info = env.reset(seed=seed, options=options)
obs_dict, state, reward_dict, terminated_dict, truncated_dict, info = env.step(action_dict)
```

`reward_dict` contains equal `idc` and `bess` values. Both done dictionaries contain `idc`, `bess`, and `__all__`, with each value copied from the corresponding legacy flag.

## 6. Time semantics and read-only state

Reset publishes only initial-time information. Because legacy reset info omits initialized `bess_soc` and `bess_energy_kWh`, the wrapper reads those two existing attributes from `GridCoupledEnv.env`. The four transition power fields are zero at reset. No other lower-layer state is used to construct supplemental values.

After `step(action_t)` completes, `_last_info`, `_last_raw_observation`, and `_last_flat_action` are updated. The returned observation/state are for the next decision and may include OPF/LMP/MEF, SOC, and power results produced by `action_t`. Those results are never available while selecting `action_t`.

On the terminal transition, the returned state retains the legacy terminal-observation convention and must not be used to select another action.

## 7. Invariance and future bridge

The outer layer does not alter the physical environment, reward, Safe cost, BESS mapping, task model, OPF/LMP/MEF, or training parameters. A future HARL bridge will need to translate this dictionary API into HARL's agent ordering, shared-observation tensors, availability masks, vectorized batching, and rollout-buffer conventions. That bridge is not implemented here.
