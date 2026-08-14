"""Two-case diagnostic validation of an approximately 25 MW IDC on IEEE-14 bus 9.

Only the current approximately 1 MW scale and one analytically calibrated
25 MW peak scale are evaluated. Formal configuration dictionaries and the
stored IEEE-14 network are never mutated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config_ultimate import (  # noqa: E402
    GRID_CONFIG,
    GRID_REWARD_CONFIG,
    GRID_SCENARIO_CONFIG,
    IDC_SCALE_CONFIG,
)
from configs.experiment_cases import get_experiment_case  # noqa: E402
from data_io.data_loader import build_external_series_from_config  # noqa: E402
from env_wrappers import GridCoupledEnv  # noqa: E402
from envs.idc_price_env import IDCPriceEnv20D  # noqa: E402
from grid_model import get_bus_index_by_ieee_number, load_ieee14_case  # noqa: E402
from grid_model.grid_case import OPFResult  # noqa: E402
from grid_model.opf_solver import solve_ac_opf  # noqa: E402


SEED = 2026
TARGET_PEAK_MW = 25.0
FIXED_ACTION_VALUE = 0.5


@dataclass(frozen=True)
class SiteHour:
    hour: int
    grid_load_scale: float
    p_it_mw: float
    p_cooling_mw: float
    p_others_mw: float
    p_idc_mw: float
    pv_available_mw: float
    pv_used_mw: float
    bess_charge_mw: float
    bess_discharge_mw: float
    p_bus_net_mw: float


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_values(values: Iterable[Any]) -> list[float]:
    return [number for number in (finite(value) for value in values) if number is not None]


def minimum(values: Iterable[Any]) -> float | None:
    numbers = finite_values(values)
    return min(numbers) if numbers else None


def maximum(values: Iterable[Any]) -> float | None:
    numbers = finite_values(values)
    return max(numbers) if numbers else None


def mean(values: Iterable[Any]) -> float | None:
    numbers = finite_values(values)
    return float(np.mean(numbers)) if numbers else None


def _make_site_env(server_group_size: int, seed: int) -> GridCoupledEnv:
    """Build a private diagnostic env while holding PV/BESS at current scale."""

    experiment = get_experiment_case("main")
    env_config = deepcopy(experiment["env_config"])
    reward_config = deepcopy(experiment["reward_config"])
    data_config = deepcopy(experiment["data_config"])
    scale = deepcopy(IDC_SCALE_CONFIG)
    scale["enable_server_group_model"] = True
    scale["num_server_groups"] = 20
    scale["server_group_size"] = int(server_group_size)
    scale["task_workload_scale"] = float(server_group_size)
    # Deliberately retain today's physical microgrid, not the new fleet scale.
    scale["bess_scale_factor"] = float(IDC_SCALE_CONFIG["bess_scale_factor"])
    scale["scale_bess_with_idc"] = bool(IDC_SCALE_CONFIG["scale_bess_with_idc"])

    forecast_seed = seed + int(env_config.get("task_forecast_seed_offset", 300000))
    kwargs = {
        **env_config,
        **reward_config,
        **scale,
        **build_external_series_from_config(data_config, env_config["horizon"]),
        "server_seed": seed,
        "task_seed": seed,
        "forecast_seed": forecast_seed,
    }
    base = IDCPriceEnv20D(**kwargs)
    wrapper = GridCoupledEnv(
        base,
        deepcopy(GRID_CONFIG),
        deepcopy(GRID_REWARD_CONFIG),
        deepcopy(GRID_SCENARIO_CONFIG),
        grid_cache_config={"enable_grid_cache": False},
    )
    wrapper.grid_enabled = False
    wrapper.use_mef = False
    return wrapper


def collect_site_trace(server_group_size: int, seed: int = SEED) -> tuple[list[SiteHour], dict[str, Any]]:
    env = _make_site_env(server_group_size, seed)
    action = np.full(env.action_space.shape, FIXED_ACTION_VALUE, dtype=np.float32)
    env.reset(seed=seed)
    base = env.env
    static = {
        "server_group_size": int(base.server_group_size),
        "num_server_groups": int(base.num_server_groups),
        "effective_total_server_count": int(base.effective_total_server_count),
        "task_workload_scale": float(base.task_workload_scale),
        "p_others_mw": float(base.model.P_others) / 1e6,
        "fixed_it_loss_mw": float(base.model.delta_P_loss) / 1e6,
        "bess_capacity_mwh": float(base.bess_capacity_kWh) / 1000.0,
        "bess_charge_power_max_mw": float(base.bess_charge_power_max_kW) / 1000.0,
        "bess_discharge_power_max_mw": float(base.bess_discharge_power_max_kW) / 1000.0,
        "pv_configured_peak_mw": float(max(base.pv_t)) / 1000.0,
        "action_shape": list(env.action_space.shape),
        "action_value": FIXED_ACTION_VALUE,
    }
    trace: list[SiteHour] = []
    try:
        for hour in range(24):
            load_scale = float(env.grid_load_scale_t[hour % len(env.grid_load_scale_t)])
            _, _, terminated, truncated, info = env.step(action)
            trace.append(
                SiteHour(
                    hour=hour,
                    grid_load_scale=load_scale,
                    p_it_mw=float(info["P_IT"]) / 1e6,
                    p_cooling_mw=float(info["P_cooling"]) / 1e6,
                    p_others_mw=float(base.model.P_others) / 1e6,
                    p_idc_mw=float(info["P_IDC_kW"]) / 1000.0,
                    pv_available_mw=float(info["pv_available_kW"]) / 1000.0,
                    pv_used_mw=float(info["pv_used_kW"]) / 1000.0,
                    bess_charge_mw=float(info["bess_charge_power_kW"]) / 1000.0,
                    bess_discharge_mw=float(info["bess_discharge_power_kW"]) / 1000.0,
                    p_bus_net_mw=max(float(info["P_bus_net_kW"]), 0.0) / 1000.0,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    if len(trace) != 24:
        raise RuntimeError(f"Expected 24 hours, got {len(trace)}.")
    return trace, static


def summarize_site(trace: list[SiteHour]) -> dict[str, Any]:
    fields = ("p_it_mw", "p_cooling_mw", "p_others_mw", "p_idc_mw", "p_bus_net_mw")
    result: dict[str, Any] = {}
    for field in fields:
        values = [getattr(hour, field) for hour in trace]
        result[f"{field}_min"] = min(values)
        result[f"{field}_mean"] = float(np.mean(values))
        result[f"{field}_max"] = max(values)
    result["pv_available_peak_mw"] = max(hour.pv_available_mw for hour in trace)
    result["pv_used_peak_mw"] = max(hour.pv_used_mw for hour in trace)
    return result


def calibrate_group_size(
    original_trace: list[SiteHour],
    current_group_size: int,
    target_peak_mw: float = TARGET_PEAK_MW,
    fixed_it_loss_mw: float = 0.005,
) -> dict[str, Any]:
    """Analytically solve the integer fleet scale using the original peak hour."""

    peak = max(original_trace, key=lambda hour: hour.p_idc_mw)
    if peak.p_cooling_mw <= 0.0 or peak.p_it_mw <= fixed_it_loss_mw:
        raise ValueError("Original peak components cannot calibrate a fleet scale.")
    cop = peak.p_it_mw / peak.p_cooling_mw
    target_it = (target_peak_mw - peak.p_others_mw) / (1.0 + 1.0 / cop)
    scalable_it_per_group = (peak.p_it_mw - fixed_it_loss_mw) / current_group_size
    exact_group_size = (target_it - fixed_it_loss_mw) / scalable_it_per_group
    chosen = max(int(round(exact_group_size)), 1)
    predicted_it = fixed_it_loss_mw + scalable_it_per_group * chosen
    predicted_peak = predicted_it * (1.0 + 1.0 / cop) + peak.p_others_mw
    return {
        "target_convention": "fixed-policy 24h peak facility demand approximately 25 MW",
        "target_peak_mw": float(target_peak_mw),
        "calibration_hour": int(peak.hour),
        "calibration_cop": float(cop),
        "exact_group_size": float(exact_group_size),
        "chosen_integer_group_size": chosen,
        "fleet_scale_factor_vs_original": chosen / float(current_group_size),
        "predicted_peak_mw": float(predicted_peak),
    }


def _generator_rows(result: OPFResult, case_name: str, hour: int) -> list[dict[str, Any]]:
    if not result.success or result.raw_result is None:
        return []
    net = result.raw_result
    rows = []
    for kind, table_name, result_name in (
        ("ext_grid", "ext_grid", "res_ext_grid"),
        ("generator", "gen", "res_gen"),
    ):
        table = getattr(net, table_name)
        res = getattr(net, result_name)
        for index, config in table.iterrows():
            idx = int(index)
            p = finite(res.at[index, "p_mw"])
            q = finite(res.at[index, "q_mvar"])
            pmin = finite(config.get("min_p_mw"))
            pmax = finite(config.get("max_p_mw"))
            qmin = finite(config.get("min_q_mvar"))
            qmax = finite(config.get("max_q_mvar"))
            p_lower = None if p is None or pmin is None else p - pmin
            p_upper = None if p is None or pmax is None else pmax - p
            q_lower = None if q is None or qmin is None else q - qmin
            q_upper = None if q is None or qmax is None else qmax - q
            p_util = (
                None
                if p is None or pmin is None or pmax is None or pmax <= pmin
                else (p - pmin) / (pmax - pmin)
            )
            q_util = (
                None
                if q is None or qmin is None or qmax is None or qmax <= qmin
                else (q - qmin) / (qmax - qmin)
            )
            rows.append(
                {
                    "case": case_name,
                    "hour": hour,
                    "element": kind,
                    "element_index": idx,
                    "element_id": f"{kind}:{idx}",
                    "bus_index": int(config.bus),
                    "p_mw": p,
                    "p_min_mw": pmin,
                    "p_max_mw": pmax,
                    "p_lower_headroom_mw": p_lower,
                    "p_upper_headroom_mw": p_upper,
                    "p_upper_utilization": p_util,
                    "q_mvar": q,
                    "q_min_mvar": qmin,
                    "q_max_mvar": qmax,
                    "q_lower_headroom_mvar": q_lower,
                    "q_upper_headroom_mvar": q_upper,
                    "q_upper_utilization": q_util,
                    "q_nearest_headroom_mvar": minimum((q_lower, q_upper)),
                }
            )
    return rows


def _opf_hour(result: OPFResult, case_name: str, site: SiteHour) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case": case_name,
        "hour": site.hour,
        "opf_success": bool(result.success),
        "opf_message": str(result.message),
        "p_idc_mw": site.p_idc_mw,
        "p_bus_net_mw": site.p_bus_net_mw,
        "grid_load_scale": site.grid_load_scale,
    }
    if not result.success or result.raw_result is None:
        return row
    net = result.raw_result
    lower = {
        bus: value - result.bus_voltage_min_pu[bus]
        for bus, value in result.bus_voltage_pu.items()
        if finite(value) is not None
    }
    upper = {
        bus: result.bus_voltage_max_pu[bus] - value
        for bus, value in result.bus_voltage_pu.items()
        if finite(value) is not None
    }
    critical_bus = min(lower, key=lower.get) if lower else None
    idc_bus = 8
    line_values = finite_values(result.line_loading_percent.values())
    trafo_values = finite_values(result.transformer_loading_percent.values())
    lmp_values = finite_values(result.lmp_by_bus.values())
    ext_p = finite(net.res_ext_grid.p_mw.sum()) if len(net.res_ext_grid) else None
    row.update(
        {
            "min_voltage_pu": minimum(result.bus_voltage_pu.values()),
            "max_voltage_pu": maximum(result.bus_voltage_pu.values()),
            "minimum_lower_voltage_margin_pu": minimum(lower.values()),
            "minimum_upper_voltage_margin_pu": minimum(upper.values()),
            "voltage_violation_count": sum(value < -1e-7 for value in lower.values())
            + sum(value < -1e-7 for value in upper.values()),
            "critical_lower_margin_bus_index": critical_bus,
            "idc_bus_voltage_pu": finite(result.bus_voltage_pu.get(idc_bus)),
            "idc_bus_lmp": finite(result.lmp_by_bus.get(idc_bus)),
            "lmp_min": min(lmp_values) if lmp_values else None,
            "lmp_max": max(lmp_values) if lmp_values else None,
            "lmp_spread": max(lmp_values) - min(lmp_values) if lmp_values else None,
            "network_loss_mw": finite(result.network_loss_mw),
            "ext_grid_import_mw": ext_p,
            "max_line_loading_percent": max(line_values) if line_values else None,
            "max_transformer_loading_percent": max(trafo_values) if trafo_values else None,
            "bus_voltage_by_index": {
                str(bus): finite(value) for bus, value in result.bus_voltage_pu.items()
            },
        }
    )
    return row


def summarize_opf(
    case_name: str,
    site_trace: list[SiteHour],
    opf_hours: list[dict[str, Any]],
    generator_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    success = [row for row in opf_hours if row["opf_success"]]
    capability: dict[str, dict[str, Any]] = {}
    for element_id in sorted({row["element_id"] for row in generator_rows}):
        rows = [row for row in generator_rows if row["element_id"] == element_id]
        capability[element_id] = {
            "element": rows[0]["element"],
            "bus_index": rows[0]["bus_index"],
            "p_min_mw": rows[0]["p_min_mw"],
            "p_max_mw": rows[0]["p_max_mw"],
            "p_observed_min_mw": minimum(row["p_mw"] for row in rows),
            "p_observed_max_mw": maximum(row["p_mw"] for row in rows),
            "minimum_p_lower_headroom_mw": minimum(row["p_lower_headroom_mw"] for row in rows),
            "minimum_p_upper_headroom_mw": minimum(row["p_upper_headroom_mw"] for row in rows),
            "maximum_p_upper_utilization": maximum(row["p_upper_utilization"] for row in rows),
            "q_min_mvar": rows[0]["q_min_mvar"],
            "q_max_mvar": rows[0]["q_max_mvar"],
            "q_observed_min_mvar": minimum(row["q_mvar"] for row in rows),
            "q_observed_max_mvar": maximum(row["q_mvar"] for row in rows),
            "minimum_q_lower_headroom_mvar": minimum(row["q_lower_headroom_mvar"] for row in rows),
            "minimum_q_upper_headroom_mvar": minimum(row["q_upper_headroom_mvar"] for row in rows),
            "minimum_q_nearest_headroom_mvar": minimum(row["q_nearest_headroom_mvar"] for row in rows),
            "maximum_q_upper_utilization": maximum(row["q_upper_utilization"] for row in rows),
        }
    return {
        "case": case_name,
        **summarize_site(site_trace),
        "opf_success_count": len(success),
        "opf_failure_count": len(opf_hours) - len(success),
        "opf_failure_hours": [row["hour"] for row in opf_hours if not row["opf_success"]],
        "min_voltage_pu": minimum(row.get("min_voltage_pu") for row in success),
        "max_voltage_pu": maximum(row.get("max_voltage_pu") for row in success),
        "minimum_lower_voltage_margin_pu": minimum(
            row.get("minimum_lower_voltage_margin_pu") for row in success
        ),
        "minimum_upper_voltage_margin_pu": minimum(
            row.get("minimum_upper_voltage_margin_pu") for row in success
        ),
        "voltage_violation_count": sum(int(row.get("voltage_violation_count", 0)) for row in success),
        "critical_lower_margin_bus_index": min(
            success,
            key=lambda row: row.get("minimum_lower_voltage_margin_pu", math.inf),
        ).get("critical_lower_margin_bus_index") if success else None,
        "idc_bus_voltage_min_pu": minimum(row.get("idc_bus_voltage_pu") for row in success),
        "idc_bus_voltage_mean_pu": mean(row.get("idc_bus_voltage_pu") for row in success),
        "idc_bus_voltage_max_pu": maximum(row.get("idc_bus_voltage_pu") for row in success),
        "idc_bus_lmp_min": minimum(row.get("idc_bus_lmp") for row in success),
        "idc_bus_lmp_mean": mean(row.get("idc_bus_lmp") for row in success),
        "idc_bus_lmp_max": maximum(row.get("idc_bus_lmp") for row in success),
        "lmp_spread_max": maximum(row.get("lmp_spread") for row in success),
        "network_loss_mean_mw": mean(row.get("network_loss_mw") for row in success),
        "network_loss_max_mw": maximum(row.get("network_loss_mw") for row in success),
        "peak_ext_grid_import_mw": maximum(row.get("ext_grid_import_mw") for row in success),
        "max_line_loading_percent": maximum(row.get("max_line_loading_percent") for row in success),
        "max_transformer_loading_percent": maximum(
            row.get("max_transformer_loading_percent") for row in success
        ),
        "maximum_source_p_upper_utilization": maximum(
            row.get("p_upper_utilization") for row in generator_rows
        ),
        "minimum_source_p_upper_headroom_mw": minimum(
            row.get("p_upper_headroom_mw") for row in generator_rows
        ),
        "maximum_source_q_upper_utilization": maximum(
            row.get("q_upper_utilization") for row in generator_rows
        ),
        "minimum_source_q_upper_headroom_mvar": minimum(
            row.get("q_upper_headroom_mvar") for row in generator_rows
        ),
        "minimum_source_q_nearest_headroom_mvar": minimum(
            row.get("q_nearest_headroom_mvar") for row in generator_rows
        ),
        "capability_by_element": capability,
    }


def run_opf_case(
    case_name: str,
    site_trace: list[SiteHour],
    solver: Callable[..., OPFResult] = solve_ac_opf,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grid_case = load_ieee14_case()
    idc_bus = get_bus_index_by_ieee_number(grid_case, 9)
    opf_hours: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    for site in site_trace:
        result = solver(
            grid_case,
            idc_bus_id=idc_bus,
            idc_load_mw=site.p_bus_net_mw,
            load_scale=site.grid_load_scale,
        )
        opf_hours.append(_opf_hour(result, case_name, site))
        generator_rows.extend(_generator_rows(result, case_name, site.hour))
    return summarize_opf(case_name, site_trace, opf_hours, generator_rows), opf_hours, generator_rows


def bus_voltage_rows(opf_hours: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for hour in opf_hours:
        for bus, value in hour.get("bus_voltage_by_index", {}).items():
            rows.append(
                {
                    "case": hour["case"],
                    "hour": hour["hour"],
                    "bus_index": int(bus),
                    "voltage_pu": value,
                }
            )
    return rows


def comparison_rows(original: dict[str, Any], scaled: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = (
        "p_idc_mw_max",
        "p_bus_net_mw_max",
        "opf_success_count",
        "min_voltage_pu",
        "minimum_lower_voltage_margin_pu",
        "minimum_upper_voltage_margin_pu",
        "idc_bus_voltage_min_pu",
        "maximum_source_q_upper_utilization",
        "minimum_source_q_upper_headroom_mvar",
        "minimum_source_q_nearest_headroom_mvar",
        "maximum_source_p_upper_utilization",
        "minimum_source_p_upper_headroom_mw",
        "network_loss_mean_mw",
        "peak_ext_grid_import_mw",
        "lmp_spread_max",
        "idc_bus_lmp_mean",
        "max_line_loading_percent",
        "max_transformer_loading_percent",
    )
    rows = []
    for metric in metrics:
        before = original.get(metric)
        after = scaled.get(metric)
        change = after - before if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
        rows.append({"metric": metric, "original": before, "idc_25mw": after, "change": change})
    return rows


def validate_no_nonfinite(payload: Any, path: str = "root") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            validate_no_nonfinite(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            validate_no_nonfinite(value, f"{path}[{index}]")
    elif isinstance(payload, (float, np.floating)) and not math.isfinite(float(payload)):
        raise ValueError(f"Non-finite value at {path}: {payload}")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
        for row in rows
    ]
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def run(output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_size = int(IDC_SCALE_CONFIG["server_group_size"])
    original_trace, original_static = collect_site_trace(original_size, seed)
    calibration = calibrate_group_size(
        original_trace,
        original_size,
        fixed_it_loss_mw=original_static["fixed_it_loss_mw"],
    )
    scaled_size = int(calibration["chosen_integer_group_size"])
    scaled_trace, scaled_static = collect_site_trace(scaled_size, seed)

    original_summary, original_opf, original_generators = run_opf_case("original_approx_1mw", original_trace)
    scaled_summary, scaled_opf, scaled_generators = run_opf_case("idc_25mw_peak", scaled_trace)
    comparison = comparison_rows(original_summary, scaled_summary)
    calibration["observed_peak_mw"] = scaled_summary["p_idc_mw_max"]
    calibration["observed_peak_error_mw"] = scaled_summary["p_idc_mw_max"] - TARGET_PEAK_MW
    calibration["original_effective_server_count"] = original_static["effective_total_server_count"]
    calibration["scaled_effective_server_count"] = scaled_static["effective_total_server_count"]

    payload = {
        "metadata": {
            "seed": seed,
            "background_multiplier": 1.0,
            "idc_ieee_bus_number": 9,
            "fixed_action_value": FIXED_ACTION_VALUE,
            "cases": ["original_approx_1mw", "idc_25mw_peak"],
            "branch_rating_note": "9900 MVA ratings make loading a response metric, not a validated safety constraint.",
        },
        "power_definition": "P_IDC = P_IT + P_cooling + P_others; P_cooling = P_IT / COP",
        "calibration": calibration,
        "original_static": original_static,
        "scaled_static": scaled_static,
        "original_summary": original_summary,
        "scaled_summary": scaled_summary,
        "comparison": comparison,
        "site_hours": [
            {"case": "original_approx_1mw", **asdict(row)} for row in original_trace
        ] + [{"case": "idc_25mw_peak", **asdict(row)} for row in scaled_trace],
        "opf_hours": original_opf + scaled_opf,
        "generator_hours": original_generators + scaled_generators,
    }
    validate_no_nonfinite(payload)
    write_json(output_dir / "idc_25mw_acopf_validation.json", payload)
    write_csv(output_dir / "case_comparison.csv", comparison)
    write_csv(output_dir / "site_power_trajectories.csv", payload["site_hours"])
    write_csv(output_dir / "opf_hourly.csv", payload["opf_hours"])
    write_csv(output_dir / "generator_capability_hourly.csv", payload["generator_hours"])
    write_csv(output_dir / "bus_voltage_hourly.csv", bus_voltage_rows(payload["opf_hours"]))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/idc_25mw_acopf_validation"))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.output_dir, seed=args.seed)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "server_group_size": result["calibration"]["chosen_integer_group_size"],
                "observed_peak_mw": result["calibration"]["observed_peak_mw"],
                "opf_success_25mw": result["scaled_summary"]["opf_success_count"],
            },
            indent=2,
        )
    )
