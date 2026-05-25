# Grid Cache Acceleration Report

## 1. Modified Files

- `grid_model/grid_cache.py`
- `configs/config_ultimate.py`
- `env_wrappers/grid_coupled_env.py`
- `eval/eval_base.py`
- `scripts/smoke_test_grid_coupled_env.py`
- `scripts/smoke_test_parallel_sampling.py`
- `scripts/benchmark_grid_cache.py`
- `train/train_ppo_ultimate.py`

`train/train_ppo_ultimate.py` was changed only for cache config propagation, cache statistics in the random VecEnv smoke helper, and a minimal VecEnv fallback for environments that do not have Stable-Baselines3 installed.

## 2. Why OPF/MEF Cache Is Needed

`GridCoupledEnv.step()` previously solved one AC OPF for grid feedback and then MEF solved base/plus/minus OPFs again. This makes random sampling and PPO rollout collection slow, especially under `SubprocVecEnv`. The cache avoids repeating identical or near-identical grid solves after load and scenario values are binned.

## 3. Cache Key Design

OPF keys include:

- `opf_mode`
- IDC pandapower bus index
- hour
- binned `grid_load_scale`
- binned IDC load MW

MEF keys include the same fields plus `delta_p_mw`.

Both keys round floating values to fixed precision after binning, avoiding keys such as `1.2000000001`.

## 4. Why 0.1 MW Binning

The default IDC load bin is `0.1 MW`. It is coarse enough to merge many nearby random-action load states while remaining small relative to the MW-scale IDC/grid perturbations used in training. `grid_load_scale` is binned at `0.005`, which keeps the NEMS 24-hour scenario shape while allowing nearby scale values to reuse results.

## 5. Why Per-Worker Cache

Each `GridCoupledEnv` owns one `GridResultCache`. In `SubprocVecEnv`, every worker process has its own environment instance, so the cache is naturally per-worker. This keeps the design simple, avoids IPC overhead, and avoids lock contention during rollout collection.

## 6. Why No Cross-Process Shared Cache

This first version intentionally does not use `multiprocessing.Manager`, file locks, SQLite, or disk cache. Shared cache would add serialization, synchronization, stale-state, and failure-mode complexity. Per-worker memory cache is safer for a first acceleration pass.

## 7. Failed Results

Failed OPF or MEF results are not cached by default. This prevents temporary numerical failures or infeasible perturbations from being replayed as if they were reliable grid states. The default config sets `cache_failed_results=False`.

## 8. GRID_CACHE_CONFIG

`configs/config_ultimate.py` now defines:

```python
GRID_CACHE_CONFIG = {
    "enable_grid_cache": True,
    "cache_opf": True,
    "cache_mef": True,
    "cache_load_bin_mw": 0.1,
    "cache_load_scale_bin": 0.005,
    "cache_max_size": 50000,
    "cache_clear_on_reset": False,
    "cache_scope": "per_worker",
    "cache_failed_results": False,
    "cache_verbose": False,
}
```

`cache_clear_on_reset=False` allows reuse across episodes. `cache_scope="per_worker"` is documentation of the design; no shared manager or disk state is created.

## 9. GridCoupledEnv Integration

`GridCoupledEnv` creates a `GridResultCache` from `GRID_CACHE_CONFIG`, unless an explicit `grid_cache_config` override is passed. The wrapper only:

- asks the cache module to build keys;
- checks OPF/MEF hits;
- calls the original solver/calculator on misses;
- stores only successful results;
- injects cache statistics into `info`.

OPF solver logic and MEF math are unchanged.

## 10. Info, Eval, and Smoke Fields

Each step now includes:

- `grid_cache_enabled`
- `grid_opf_cache_enabled`
- `grid_mef_cache_enabled`
- `grid_opf_cache_hit`
- `grid_mef_cache_hit`
- `grid_opf_cache_hit_count`
- `grid_opf_cache_miss_count`
- `grid_opf_cache_hit_rate`
- `grid_mef_cache_hit_count`
- `grid_mef_cache_miss_count`
- `grid_mef_cache_hit_rate`
- `grid_cache_opf_size`
- `grid_cache_mef_size`
- `grid_cache_load_bin_mw`
- `grid_cache_load_scale_bin`

`eval/eval_base.py` adds cache columns to hourly rows and summary metrics. The smoke scripts print cache enabled state, hit rates, sizes, and bins.

## 11. benchmark_grid_cache.py

New command:

```bash
python -m scripts.benchmark_grid_cache --n-envs 1 --vec-env dummy --steps 240 --cache both
python -m scripts.benchmark_grid_cache --n-envs 6 --vec-env subproc --steps 240 --cache both
```

Supported arguments:

- `--n-envs`
- `--vec-env dummy/subproc`
- `--steps`
- `--seed`
- `--cache on/off/both`
- `--load-bin-mw`
- `--load-scale-bin`
- `--cpu-threads-per-worker`

The script runs random actions only. It does not train PPO or create a PPO model.

## 12. Benchmark Results

Codex bundled Python does not include `stable_baselines3`. Therefore the current environment could complete `py_compile`, the single-environment GridCoupledEnv smoke test, and a 1-env benchmark using the lightweight fallback VecEnv. The 6-env benchmark was stopped and should be run by the user in the local `idc_ppo` environment.

Observed in Codex temporary environment:

| n_envs | vec_env | cache | total_seconds | steps_per_second | OPF success | MEF success | OPF hit rate | MEF hit rate | reward_mismatch_count | safe_cost |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dummy fallback | off | 211.099 | 1.137 | 240/240 | 240/240 | 0.0000 | 0.0000 | 0 | 0.000 |
| 1 | dummy fallback | on | 213.087 | 1.126 | 240/240 | 240/240 | 0.2520 | 0.2520 | 0 | 0.000 |
| 6 | subproc | off/on | not run in Codex | not run in Codex | not run in Codex | not run in Codex | not run in Codex | not run in Codex | not run in Codex | not run in Codex |

The 1-env 240-step cache-on run observed about 25.2% OPF and MEF hit rate. Under this random-action distribution, that hit rate was not enough to produce a visible speedup in the Codex temporary environment.

Recommended local benchmark:

```bash
python -m scripts.benchmark_grid_cache --n-envs 1 --vec-env dummy --steps 240 --cache both
python -m scripts.benchmark_grid_cache --n-envs 6 --vec-env subproc --steps 240 --cache both
```

Run these in the local `idc_ppo` environment where `stable_baselines3` is installed.

## 13. Stable-Baselines3 and PPO Path

When `stable_baselines3` is installed, `build_vec_env()` still uses the original SB3 `DummyVecEnv` and `SubprocVecEnv`. The fallback VecEnv is used only when SB3 is missing, so Codex can run minimum smoke/benchmark checks in a limited Python environment. Formal PPO training still imports `stable_baselines3.PPO`; if SB3 is missing, training exits instead of silently using the fallback for PPO.

## 14. Invariants

- Reward changed: no.
- Safe cost changed: no.
- OPF/MEF math changed: no.
- Server cluster model changed: no.
- BESS scaling changed: no.
- NEMS data processing changed: no.
- Observation dimension changed: no, still 264.
- Action dimension changed: no, still 23.
- PPO training run: no.
- `algorithms/` modified: no.

## 15. Verification Completed

Completed with Codex bundled Python:

```bash
python -m py_compile grid_model/grid_cache.py env_wrappers/grid_coupled_env.py eval/eval_base.py scripts/smoke_test_grid_coupled_env.py scripts/smoke_test_parallel_sampling.py scripts/benchmark_grid_cache.py train/train_ppo_ultimate.py
python -m scripts.smoke_test_grid_coupled_env
python -m scripts.benchmark_grid_cache --n-envs 1 --vec-env dummy --steps 240 --cache both
```

Single-env smoke observed:

- `obs_dim=264`
- `action_dim=23`
- OPF success `24/24`
- MEF success `24/24`
- `reward_mismatch_count=0`
- `total_safe_cost=0`
- cache fields printed correctly

## 16. Follow-Up Suggestions

- Benchmark `0.05 / 0.1 / 0.2 MW` load bins.
- Run final evaluation with cache disabled if exact OPF/MEF semantics are required.
- Build an offline lookup table after the bin size is validated.
- Add a future `fast_grid_mode` for large training sweeps.
- Continue grid reward ablation and SAFE RL after cache behavior is validated.
