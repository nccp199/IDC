# Reward Final Audit Report

Date: 2026-05-24

Scope: training-preparation reward audit only. No reward code, weights, grid logic, BESS logic, SAFE RL, MAPPO, or PPO training was modified or run.

## 1. Checked Files

Primary files checked:

- `configs/config_ultimate.py`
- `envs/idc_price_env.py`
- `env_wrappers/grid_coupled_env.py`
- `idc_model/task_model.py`
- `idc_model/power_model.py`
- `eval/eval_base.py`
- `scripts/smoke_test_grid_coupled_env.py`
- `train/train_ppo_ultimate.py`

Additional reward-adjacent files checked:

- `grid_model/grid_metrics.py`
- `grid_model/opf_solver.py`
- `grid_model/ieee14_loader.py`

## 2. Current Base IDC Reward Composition

The base IDC reward is computed in `IDCPriceEnv20D.step()` before `GridCoupledEnv` applies any optional grid reward penalty.

Positive reward terms:

- `r_done = reward_done_weight * completed_work / queue_ref`
- `r_finished_task = reward_finished_task_weight * newly_finished_count / task_count`
- `r_priority_finish = reward_priority_finish_weight * newly_finished_priority_sum / (5 * task_count)`

Penalty terms:

- `r_cost = -reward_cost_weight * hourly_cost / cost_ref`
- `r_carbon = -reward_carbon_weight * carbon_emission / carbon_ref`
- `r_queue = -reward_queue_weight * backlog_work / queue_ref`
- `r_queue_overflow = -reward_queue_overflow_weight * overflow_work / queue_ref`
- `r_urgent_backlog = -reward_urgent_backlog_weight * urgent_backlog_work / queue_ref`
- `r_waiting = -reward_waiting_weight * avg_waiting_pressure / horizon`
- `r_deadline = -reward_deadline_weight * new_deadline_miss_count / task_count`
- `r_sla = -reward_sla_weight * sla_penalty / sla_ref`
- `r_unused = -reward_unused_capacity_weight * unused_capacity / queue_ref`
- `r_peak_load` / `r_grid_peak = -reward_grid_peak_weight * grid_peak_excess_kW / peak_power_ref_kW`
- `r_pause = -reward_pause_weight * pause_count_this_step / task_count`
- `r_resume = -reward_resume_weight * resume_count_this_step / task_count`
- `r_non_interruptible = -reward_non_interruptible_weight * interruption_count / task_count`
- `r_load_smooth = -reward_load_smooth_weight * load_change`
- `r_action_smooth = -reward_action_smooth_weight * action_change`
- `r_bess_degradation = -reward_bess_degradation_weight * bess_degradation_cost / cost_ref`
- `r_bess_invalid_action = -reward_bess_invalid_action_weight * invalid_bess_action / bess_power_ref`
- terminal `r_final_queue = -reward_final_queue_weight * final_backlog / queue_ref`
- terminal `r_soc_final = -reward_soc_final_weight * max(abs(final_soc - target_soc) - tolerance, 0)`

`r_grid_peak` is logged as an alias, but only `r_peak_load` is included in the reward sum, so this is not a duplicate reward penalty.

## 3. Scaling Audit

Current scaling:

- `server_group_size=100`
- `num_server_groups=20`
- effective server count = `2000`
- `task_workload_scale=100`
- `bess_scale_factor=100`
- `scale_bess_with_idc=True`

The key reward normalizers are scaled consistently:

- `initial_Q`, `lambda_ref`, `queue_ref`, `queue_capacity_ref`, and `sla_ref` are scaled by `task_workload_scale`.
- `cost_ref`, `carbon_ref`, `peak_power_threshold_kW`, `peak_power_ref_kW`, and `grid_power_limit_kW` are scaled by IDC power scale, currently `server_group_size`.
- BESS capacity and charge/discharge power are scaled by `bess_scale_factor`.
- Task workload generation uses base IDC capacity times `task_workload_scale`, while server capacity and power are scaled by `server_group_size`.

Result: no obvious scaling mismatch was found in the base reward path. The random-policy smoke test is still dominated by backlog and peak penalties, but that is consistent with an incomplete random schedule, not a scaling bug.

Observed component sums over the 24-hour smoke episode:

| Component | Sum |
|---|---:|
| `r_done` | `+2.9923` |
| `r_finished_task` | `+0.9194` |
| `r_priority_finish` | `+0.1920` |
| `r_cost` | `-1.1642` |
| `r_carbon` | `-3.5243` |
| `r_queue` | `-10.2297` |
| `r_urgent_backlog` | `-3.0015` |
| `r_waiting` | `-0.9992` |
| `r_deadline` | `-0.5419` |
| `r_sla` | `-0.0375` |
| `r_peak_load` | `-5.4085` |
| `r_bess_degradation` | `-0.0771` |
| `r_bess_invalid_action` | `-0.2330` |
| terminal `r_final_queue` | `-2.1449` |
| terminal `r_soc_final` | `-0.3302` |

The base reward sum matched the sum of reward components with max absolute numerical error `8.9e-16`.

## 4. Grid Reward Ablation Composition

Current grid reward ablation is implemented as:

- LMP cost: `grid_lmp * grid_energy_mwh`
- MEF carbon: `grid_mef_plus * grid_energy_mwh`
- security: `safe_violation_cost`

Normalization:

- `grid_lmp_cost_norm = grid_lmp_cost / lmp_cost_ref`
- `grid_mef_carbon_norm = grid_mef_carbon / mef_carbon_ref`
- `grid_safe_violation_norm = safe_violation_cost / safe_violation_ref`

Mode behavior:

- `none`: no penalty
- `lmp`: LMP term only
- `mef`: MEF term only
- `security`: strict safe violation term only
- `lmp_mef`: LMP + MEF
- `full`: LMP + MEF + safe violation

The implementation uses OPF `grid_lmp`, not `grid_reference_usep`. The security reward term uses `safe_violation_cost`, not `grid_security_penalty`.

## 5. Default Grid Reward Behavior

Default config audit:

- `GRID_CONFIG["enable_grid_reward"] = False`
- `GRID_REWARD_CONFIG["enable_grid_reward"] = False`
- `GRID_REWARD_CONFIG["grid_reward_mode"] = "none"`
- `grid_lmp_cost_weight = 0.0`
- `grid_mef_carbon_weight = 0.0`
- `grid_safe_violation_weight = 0.0`

Smoke test results:

- `grid_reward_enabled=False`
- `grid_reward_mode=none`
- `total_grid_reward_penalty=0`
- `total_grid_lmp_cost_penalty=0`
- `total_grid_mef_carbon_penalty=0`
- `total_grid_safe_violation_penalty=0`
- `reward_mismatch_count=0`
- `base_reward == grid_adjusted_reward` at all 24 steps

Conclusion: default grid reward does not change the base reward.

## 6. Smoke Test Reward Ranges

Command run:

```bash
python -m scripts.smoke_test_grid_coupled_env
```

The shell did not have `python`/`py` on PATH, so this was run with:

```text
C:\Users\bulio\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

Main results:

- `obs_dim=264`
- `action_dim=23`
- OPF success: `24/24`
- MEF success: `24/24`
- total reward: `-24.051955`
- per-step base reward range: `[-4.739361, 0.118469]`
- total cost: `19958.251828`
- total carbon emission: `17621.263259`
- total grid energy: `27817.779991 kWh`
- completion rate: `0.455649`
- task completion rate: `0.612903`
- safe violation cost total: `0`
- reward mismatch count: `0`

BESS observed ranges:

- final BESS SOC: `0.715079`
- BESS SOC range: `[0.125300, 0.715079]`
- max charge power: `1686.508 kW`
- max discharge power: `1014.658 kW`
- total charge: `13234.008 kWh`
- total discharge: `9900.438 kWh`
- total degradation cost: `462.689`
- max hourly degradation cost: `33.730`
- `r_bess_degradation` total: `-0.0771`
- `r_bess_invalid_action` total: `-0.2330`

No NaN/inf was observed in the audited per-step reward/grid/BESS fields.

Parallel smoke commands:

```bash
python -m scripts.smoke_test_parallel_sampling --n-envs 2 --vec-env dummy --steps 48
python -m scripts.smoke_test_parallel_sampling --n-envs 2 --vec-env subproc --steps 48
```

In the current bundled Python, both commands stopped before VecEnv creation because `stable_baselines3` and `torch` are not installed. This is a local dependency issue in the checked interpreter, not a reward design issue. The error was explicit: `stable_baselines3 is required for DummyVecEnv/SubprocVecEnv sampling`.

## 7. BESS Reward and Dispatch Audit

BESS mechanics:

- Action maps to charge/discharge tendency through `bess_raw_action = 2 * action - 1`.
- Charge is limited by max charge power and SOC headroom.
- Discharge is limited by max discharge power, SOC floor, and current IDC load.
- No sell-back is enforced through `P_grid_kW = max(P_IDC_kW + charge - discharge, 0)`, and discharge is additionally capped by `P_IDC_kW`.
- Degradation cost is `throughput_kWh * bess_degradation_cost_per_kWh`.

Penalty scale:

- `bess_degradation_cost_per_kWh=0.02`.
- `cost_ref` is scaled to `6000`, so degradation reward penalty is small.
- In the smoke run, total `r_bess_degradation=-0.0771`, far smaller than queue, peak, cost, and carbon terms.
- Invalid action penalty is also modest: total `-0.2330`, max hourly `-0.0926`.
- Terminal SOC penalty was `-0.3302`, not large enough to dominate the episode.

Conclusion: current BESS penalties do not appear large enough to prevent high-price discharge. However, BESS arbitrage is encouraged only indirectly through reduced grid cost/carbon/peak penalties. There is no explicit low-price-charge/high-price-discharge shaping term. This is acceptable for PPO smoke training, but BESS behavior should get a later dedicated analysis after PPO starts learning.

## 8. Future Grid Reward Scale Risk

Observed raw grid ablation scales in the smoke run:

- `grid_lmp_cost_norm`: max `1.0886`, total `11.1756`
- `grid_mef_carbon_norm`: max `7.0627`, total `54.1096`
- `grid_safe_violation_norm`: total `0`
- `grid_reward_penalty_clip=10`

Risk assessment:

- LMP term with weight `1.0` is comparable to a normal per-step base reward term.
- MEF term with weight `1.0` can dominate the base reward, because per-step MEF norm reached about `7.06`.
- Full mode with weight `1.0` for LMP and MEF could apply a penalty near the same order as the worst current per-step reward, and in some cases could hit the clip.
- Security term with `safe_violation_ref=1.0` is interpretable for count-style safe violations. If OPF failure occurs, `safe_violation_opf=1`, so weight directly controls the OPF failure penalty.

Recommendation for future ablations: keep default disabled; when enabling MEF, use small weights or raise `mef_carbon_ref` to a larger value such as `500-1000` if the goal is a reward contribution comparable to existing cost/carbon terms. Consider lowering `grid_reward_penalty_clip` to `1-3` for early PPO experiments if grid reward is enabled.

No immediate code change is recommended before PPO smoke training.

## 9. Duplicate Penalty Audit

Default configuration:

- No duplicate grid LMP penalty, because grid reward is disabled.
- No duplicate grid MEF penalty, because grid reward is disabled.
- Safe violation is logged but not included in base IDC reward.
- `grid_security_penalty` is not used for reward.
- `safe_cost_total` equals `safe_violation_cost` and is only logged/evaluated.
- BESS degradation appears once in the base reward.
- `r_grid_peak` and `r_peak_load` are aliases; only `r_peak_load` enters the reward sum.

Future ablation caveat:

- If `grid_reward_mode=lmp`, then electricity cost is penalized both by IDC `r_cost` and by grid LMP unless the experiment intentionally studies this combined signal.
- If `grid_reward_mode=mef`, then carbon is penalized both by IDC `r_carbon` and grid MEF unless intentionally treated as marginal-grid carbon shaping.
- This is acceptable for ablation, but weights should be interpreted as additional penalties rather than replacements.

Conclusion: no duplicate reward penalty is active under the default training configuration.

## 10. Reward Sign Audit

No sign errors were found.

- Completion terms are positive.
- Cost, carbon, backlog, deadline, SLA, peak, smoothing, BESS degradation, invalid BESS action, final queue, and terminal SOC deviations are negative.
- Grid reward applies as `grid_adjusted_reward = base_reward - grid_reward_penalty`, which is directionally correct.
- No double-negative reward path was found.

## 11. Safe Cost Audit

Safe fields:

- `safe_violation_voltage = voltage_violation_count`
- `safe_violation_line = line_overload_count`
- `safe_violation_opf = 0 if OPF succeeds else 1`
- `safe_violation_cost = voltage + line + opf`
- `safe_cost_total = safe_violation_cost`

Smoke test:

- `total_safe_violation_voltage=0`
- `total_safe_violation_line=0`
- `total_safe_violation_opf=0`
- `total_safe_violation_cost=0`
- `total_safe_cost=0`
- safe cost mismatch count: `0`

Voltage thresholds:

- `grid_metrics.py` has fallback defaults `0.95/1.05`, but it first uses OPF result bus min/max limits extracted by `opf_solver.py`.
- `ieee14_loader.py` harmonizes generator voltage setpoints with bus limits.
- The smoke test had max voltage above `1.05` while violation count remained `0`, which indicates the fixed `1.05` threshold is not being blindly used as the actual violation criterion when bus limits are present.

Conclusion: safe cost currently has strict safety-violation semantics.

## 12. NaN / Inf Risk Audit

Reward path:

- Most reward denominators use `max(ref, 1e-6)` or task-count guards.
- BESS denominator uses `max(bess_power_ref, 1e-6)`.
- Grid reward uses `_safe_float`, `_finite_or_zero`, and positive reference guards.
- Missing or NaN grid LMP/MEF/energy values become zero for grid reward penalty.
- OPF/MEF failure does not create NaN in reward; it is converted into info flags and safe violation fields.

Observed smoke result:

- No NaN/inf in audited per-step reward, BESS, safe, or grid reward fields.

Residual reporting risk:

- `unit_task_cost`, `energy_per_task`, `idc_energy_per_task`, and `carbon_per_task` are set to `np.inf` if `total_completed_work == 0`.
- This does not affect reward calculation, but eval summary/reporting can propagate `inf` if a policy completes no work.
- This is not a PPO-smoke blocker, but a future reporting hygiene improvement would be to convert those `inf` metrics to NaN before aggregate summaries.

## 13. Immediate Change Recommendation

Do not modify reward code before PPO smoke training.

Reasons:

- Default grid reward is inactive and verified not to change reward.
- Reward signs are correct.
- Normalizers are scaled with task/power/BESS scaling.
- Safe cost is strict and not entering reward by default.
- No active duplicate reward penalty was found.
- BESS penalties are small and not obviously suppressing useful discharge.

Suggested future-only adjustments after PPO smoke:

- If MEF grid reward is enabled, start with small weights or increase `mef_carbon_ref`.
- Analyze learned BESS behavior separately, especially terminal SOC effects and whether indirect cost/carbon incentives are sufficient.
- Sanitize inf evaluation metrics for zero-completion policies.

## 14. PPO Smoke Training Readiness

Reward-wise, the environment is safe to enter a 2000-step PPO smoke training.

Recommended command:

```bash
python -m train.train_ppo_ultimate --case main --timesteps 2000 --run-name reward_smoke --n-envs 1 --vec-env auto --n-steps 128 --batch-size 64 --cpu-threads-per-worker 1
```

If using the current bundled Python checked in this audit, install or switch to an environment containing Stable-Baselines3 and Torch first. The reward audit itself does not require any reward/code change before training.

## 15. Final Conclusions

- 当前默认 reward 是否安全可用于 PPO smoke training：是
- 当前 grid_reward 默认是否未改变 base_reward：是
- 是否发现 reward 符号错误：否
- 是否发现重复惩罚：否（默认配置下）
- 是否发现严重尺度失衡：否
- BESS reward 是否需要后续专项调整：是
- 是否建议现在修改代码：否
- 是否可以进行 2000 step PPO smoke training：是（reward 角度；运行环境需有 Stable-Baselines3/Torch）
