# Grid Reward Ablation and Parallel Sampling Report

Date: 2026-05-24

## 1. Modified Files

- `configs/config_ultimate.py`
- `env_wrappers/grid_coupled_env.py`
- `eval/eval_base.py`
- `train/train_ppo_ultimate.py`
- `scripts/smoke_test_grid_coupled_env.py`
- `scripts/smoke_test_parallel_sampling.py`
- `docs/dev_reports/GRID_REWARD_PARALLEL_SAMPLING_REPORT.md`

## 2. Grid Reward Ablation Design

`GRID_REWARD_CONFIG` now defines grid reward as an opt-in ablation surface:

- `enable_grid_reward=False` by default.
- `grid_reward_mode="none"` by default.
- All three weights default to `0.0`.
- Therefore the default rollout keeps `grid_adjusted_reward == base_reward`.

Supported modes:

- `none`: no grid reward penalty.
- `lmp`: LMP energy cost only.
- `mef`: MEF marginal carbon only.
- `security`: strict safe violation cost only.
- `lmp_mef`: LMP + MEF.
- `full`: LMP + MEF + strict safe violation.

The three reward components are:

- LMP cost: `grid_lmp * (grid_energy_kWh / 1000)`, normalized by `lmp_cost_ref`.
- MEF carbon: `grid_mef_plus * (grid_energy_kWh / 1000)`, normalized by `mef_carbon_ref`.
- Safe violation: `safe_violation_cost`, normalized by `safe_violation_ref`.

The security reward term intentionally uses `safe_violation_cost`, not `grid_security_penalty`. `grid_reference_usep` remains a market reference signal only; it does not replace OPF LMP.

## 3. Reward Info and Eval Fields

`GridCoupledEnv.info` now emits these fields every step, even when grid reward is disabled:

- `grid_reward_enabled`
- `grid_reward_mode`
- `grid_lmp_cost`, `grid_lmp_cost_norm`, `grid_lmp_cost_penalty`
- `grid_mef_carbon`, `grid_mef_carbon_norm`, `grid_mef_carbon_penalty`
- `grid_safe_violation`, `grid_safe_violation_norm`, `grid_safe_violation_penalty`
- `grid_reward_penalty`
- `base_reward`
- `grid_adjusted_reward`

`eval/eval_base.py` records these fields in hourly rows and adds summary metrics:

- `total_grid_reward_penalty`
- `avg_grid_reward_penalty`
- `total_grid_lmp_cost_penalty`
- `total_grid_mef_carbon_penalty`
- `total_grid_safe_violation_penalty`
- `reward_mismatch_count`

`reward_mismatch_count` counts steps where `enable_grid_reward=False` but `grid_adjusted_reward` differs from `base_reward`.

## 4. CPU Parallel Sampling Design

`train/train_ppo_ultimate.py` now supports:

- `--n-envs` with default `1`.
- `--vec-env dummy|subproc|auto` with default `auto`.
- `--start-method` with default `spawn`.
- `--n-steps` and `--batch-size` PPO overrides.
- `--cpu-threads-per-worker` with default `1`.
- `--parallel-smoke` and `--parallel-smoke-steps`.

`auto` chooses `DummyVecEnv` for `n_envs <= 1` and `SubprocVecEnv` for `n_envs > 1`.

`DummyVecEnv` runs all environments in the main process. It is simpler and easier to debug, but does not use multiple CPU processes. `SubprocVecEnv` runs each worker in a subprocess, so OPF/MEF sampling can use multiple CPU cores.

## 5. Windows Safety Measures

- `main()` is guarded by `if __name__ == "__main__"`.
- `mp.freeze_support()` is called in both the training entry and smoke script.
- No environment or pandapower net is created at module global scope.
- Each worker creates `IDCPriceEnv20D`, `GridCoupledEnv`, and IEEE14 grid state inside its `_init()` callable.
- Workers use `seed + rank` when a base seed is provided.
- `SubprocVecEnv` uses `start_method="spawn"` by default.
- OMP/MKL/NUMEXPR/OPENBLAS thread env vars default to `1` early in `train/train_ppo_ultimate.py` and are also set inside each worker.
- VecEnv creation failures raise explicit errors instead of silently hanging.

## 6. Commands

Single environment training:

```bash
python -m train.train_ppo_ultimate --case main --timesteps 10000 --run-name smoke
```

2-env parallel smoke:

```bash
python -m scripts.smoke_test_parallel_sampling --n-envs 2 --vec-env subproc --steps 48
python -m train.train_ppo_ultimate --parallel-smoke --n-envs 2 --vec-env subproc --parallel-smoke-steps 48
```

4-env training example:

```bash
python -m train.train_ppo_ultimate --case main --n-envs 4 --vec-env subproc --n-steps 384 --batch-size 256 --timesteps 100000 --run-name ppo_4env
```

Dummy fallback:

```bash
python -m scripts.smoke_test_parallel_sampling --n-envs 2 --vec-env dummy --steps 48
```

## 7. Smoke Test Results

The shell did not have `python` or `py` on PATH, so checks were run with the Codex bundled Python:

`C:\Users\bulio\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

Compile check:

- Passed for `configs/config_ultimate.py`, `env_wrappers/grid_coupled_env.py`, `eval/eval_base.py`, `train/train_ppo_ultimate.py`, `scripts/smoke_test_parallel_sampling.py`, and `scripts/smoke_test_grid_coupled_env.py`.

`python -m scripts.smoke_test_grid_coupled_env`:

- `final_obs_dim=264`
- `action_dim=23`
- `grid_opf_success_count=24`
- `grid_opf_fail_count=0`
- `grid_mef_success_count=24`
- `grid_mef_fail_count=0`
- `total_safe_violation_cost=0`
- `total_grid_reward_penalty=0`
- `grid_reward_mode=none`
- `reward_mismatch_count=0`

`python -m scripts.smoke_test_parallel_sampling --n-envs 2 --vec-env subproc --steps 48`:

- Blocked before VecEnv creation because the available Python does not have `stable_baselines3` installed.
- Error: `stable_baselines3 is required for DummyVecEnv/SubprocVecEnv sampling.`

Dummy fallback:

- Also blocked for the same missing `stable_baselines3` dependency.
- This is an environment dependency issue, not a Subproc worker construction failure.

`python -m train.train_ppo_ultimate --parallel-smoke --n-envs 2 --vec-env subproc --parallel-smoke-steps 48`:

- Environment sanity check passed with `obs.shape=(264,)` and `action_space.shape=(23,)`.
- VecEnv creation was blocked by the same missing `stable_baselines3` dependency.

The project dependency snapshot lists `stable_baselines3==2.8.0` and `torch==2.11.0`, but the currently available bundled Python contains neither package.

## 8. Follow-Up Suggestions

- Re-run the parallel smoke commands in the project training environment with Stable-Baselines3 and Torch installed.
- Run a short PPO smoke training after VecEnv smoke passes.
- Start grid reward ablation with one nonzero weight at a time.
- Keep SAFE RL and MAPPO+CTDE as later stages after PPO/Grid reward baselines are stable.
