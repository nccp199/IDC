"""Build IEEE14 load-scale scenarios from NEMS Singapore half-hour CSV data.

The script supports both single-day RT72-style files and multi-day USEP files.
It never downloads data; place raw CSV files under:

    data/grid_scenarios/nems_singapore/raw/

Run examples:
    python -m scripts.build_grid_load_scale_from_nems
    python -m scripts.build_grid_load_scale_from_nems --input data/grid_scenarios/nems_singapore/raw/USEP_May-2026.csv --selection typical --export-all-days --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "grid_scenarios" / "nems_singapore" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "grid_scenarios" / "nems_singapore" / "processed"
REQUIRED_PERIODS = 48


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Raw NEMS CSV path. Defaults to USEP*.csv, then newest CSV in raw/.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for processed files.")
    parser.add_argument("--target-date", type=str, default=None, help="Date to process, e.g. 01-May-2026 or 2026-05-01.")
    parser.add_argument(
        "--selection",
        type=str,
        default="typical",
        choices=["first-complete", "typical", "peak-demand", "peak-price", "low-demand"],
        help="How to select one complete day when --target-date is not set.",
    )
    parser.add_argument("--export-all-days", action="store_true", help="Export hourly profiles for every complete day.")
    parser.add_argument(
        "--scale-method",
        type=str,
        default="mean",
        choices=["mean", "max", "global-mean"],
        help="Normalize selected day demand by selected day mean/max or all complete days mean.",
    )
    parser.add_argument("--hours", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean"], help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", help="Print detailed processing diagnostics.")
    args = parser.parse_args()

    DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    input_path = resolve_input_path(args.input)
    rows = read_csv_rows(input_path)
    columns = detect_columns(rows)
    records = normalize_records(rows, columns)
    records_by_date = group_records_by_date(records)
    complete_dates, incomplete_dates = classify_complete_dates(records_by_date)
    if not complete_dates:
        raise ValueError("No complete NEMS days found. A complete day must contain periods 1..48.")

    complete_daily_stats = build_daily_stats(records_by_date, complete_dates)
    selected_date = choose_selected_date(
        daily_stats=complete_daily_stats,
        target_date=args.target_date,
        selection=args.selection,
    )
    global_mean_demand = mean(stat["mean_demand"] for stat in complete_daily_stats.values())

    selected_records = records_by_date[selected_date]
    hourly_rows = aggregate_hourly_day(
        selected_records,
        source_file=input_path,
        scale_method=args.scale_method,
        selection_method=args.selection if not args.target_date else "target-date",
        global_mean_demand=global_mean_demand,
        hours=int(args.hours),
    )

    out_dir = resolve_output_dir(args.out_dir)
    output_csv = out_dir / "nems_24h_load_scale.csv"
    output_meta = out_dir / "nems_24h_load_scale_meta.json"
    write_output_csv(output_csv, hourly_rows)

    all_days_output = None
    if args.export_all_days:
        all_days_rows = build_all_days_hourly_profiles(
            records_by_date=records_by_date,
            complete_dates=complete_dates,
            source_file=input_path,
            global_mean_demand=global_mean_demand,
            hours=int(args.hours),
        )
        all_days_output = out_dir / "nems_all_days_hourly_profiles.csv"
        write_all_days_csv(all_days_output, all_days_rows)

    selected_demands = [float(row["demand_mw"]) for row in hourly_rows]
    selected_usep = [float(row["usep_sgd_per_mwh"]) for row in hourly_rows]
    metadata = {
        "source_file": str(input_path),
        "selected_date": selected_date,
        "selection_method": args.selection if not args.target_date else "target-date",
        "scale_method": args.scale_method,
        "original_rows": len(rows),
        "all_dates": sorted(records_by_date.keys()),
        "complete_dates": complete_dates,
        "incomplete_dates": incomplete_dates,
        "complete_day_count": len(complete_dates),
        "daily_mean_demand": mean(selected_demands),
        "daily_max_demand": max(selected_demands),
        "daily_mean_usep": mean(selected_usep),
        "daily_max_usep": max(selected_usep),
        "global_mean_demand": global_mean_demand,
        "output_csv": str(output_csv),
        "all_days_output_csv": str(all_days_output) if all_days_output else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if args.verbose:
        scale_values = [float(row["grid_load_scale"]) for row in hourly_rows]
        print(f"Input CSV: {input_path}")
        print(f"Detected columns: {columns}")
        print(f"All dates: {len(records_by_date)}")
        print(f"Complete dates: {len(complete_dates)}")
        print(f"Incomplete dates: {len(incomplete_dates)}")
        print(f"Selected date: {selected_date}")
        print(f"Selection method: {metadata['selection_method']}")
        print(f"Scale method: {args.scale_method}")
        print(f"Selected demand range MW: {min(selected_demands):.3f} - {max(selected_demands):.3f}")
        print(f"Selected USEP range SGD/MWh: {min(selected_usep):.3f} - {max(selected_usep):.3f}")
        print(f"grid_load_scale range: {min(scale_values):.6f} - {max(scale_values):.6f}")

    print(f"Saved processed CSV: {output_csv}")
    print(f"Saved metadata JSON: {output_meta}")
    if all_days_output:
        print(f"Saved all-days hourly CSV: {all_days_output}")


def resolve_input_path(input_arg: str | None) -> Path:
    if input_arg:
        path = Path(input_arg).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return find_default_raw_csv(DEFAULT_RAW_DIR).resolve()


def resolve_output_dir(out_dir_arg: str) -> Path:
    out_dir = Path(out_dir_arg).expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def find_default_raw_csv(raw_dir: Path) -> Path:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw NEMS directory does not exist: {raw_dir}")
    candidates = list(raw_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in raw NEMS directory: {raw_dir}")
    usep_candidates = [path for path in candidates if "usep" in path.name.lower()]
    preferred = usep_candidates if usep_candidates else candidates
    return sorted(preferred, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"NEMS CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"NEMS CSV has no data rows: {path}")
    return rows


def detect_columns(rows: list[dict[str, str]]) -> dict[str, str]:
    headers = [header for header in rows[0].keys() if header is not None]
    normalized = {header: normalize_header(header) for header in headers}

    def pick(predicate, label: str, required: bool = True) -> str | None:
        matches = [header for header, norm in normalized.items() if predicate(norm)]
        if not matches:
            if required:
                raise ValueError(f"Could not detect required NEMS column: {label}. Headers={headers}")
            return None
        return matches[0]

    columns: dict[str, str] = {
        "date": pick(lambda norm: norm in {"date", "tradingdate", "settlementdate"} or norm.endswith("date"), "Date"),
        "period": pick(lambda norm: norm in {"period", "tradingperiod", "dispatchperiod"} or norm.endswith("period"), "Period"),
        "demand": pick(lambda norm: "demand" in norm and ("mw" in norm or norm == "demand"), "Demand (MW)"),
        "usep": pick(lambda norm: norm == "usep" or norm.startswith("usep"), "USEP ($/MWh)"),
    }
    solar_column = pick(lambda norm: "solar" in norm and ("mw" in norm or norm == "solar"), "Solar", required=False)
    if solar_column is not None:
        columns["solar"] = solar_column
    return columns


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_records(rows: list[dict[str, str]], columns: dict[str, str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            record = {
                "date": parse_date(row[columns["date"]]),
                "period": int(float(clean_number(row[columns["period"]]))),
                "demand_mw": float(clean_number(row[columns["demand"]])),
                "usep_sgd_per_mwh": float(clean_number(row[columns["usep"]])),
            }
            if "solar" in columns:
                solar_raw = row.get(columns["solar"], "")
                record["solar_mw"] = float(clean_number(solar_raw)) if str(solar_raw).strip() else math.nan
        except Exception as exc:
            raise ValueError(f"Failed to parse row {row_number}: {exc}") from exc
        records.append(record)
    return records


def parse_date(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("empty date")
    text = re.sub(r"\s+", " ", text)
    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%d%B%Y",
        "%d%b%Y",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    if " " in text:
        return parse_date(text.split(" ")[0])
    raise ValueError(f"unsupported date format: {value!r}")


def clean_number(value: str) -> str:
    text = str(value).strip()
    text = text.replace(",", "").replace("$", "")
    text = re.sub(r"[^0-9eE+\-.]", "", text)
    if text in {"", ".", "-", "+"}:
        raise ValueError(f"not a numeric value: {value!r}")
    return text


def group_records_by_date(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["date"])].append(record)
    for date_value in grouped:
        grouped[date_value] = sorted(grouped[date_value], key=lambda item: int(item["period"]))
    return dict(grouped)


def classify_complete_dates(records_by_date: dict[str, list[dict[str, object]]]) -> tuple[list[str], list[str]]:
    complete_dates: list[str] = []
    incomplete_dates: list[str] = []
    required = set(range(1, REQUIRED_PERIODS + 1))
    for date_value in sorted(records_by_date):
        periods = {int(record["period"]) for record in records_by_date[date_value]}
        if required.issubset(periods):
            complete_dates.append(date_value)
        else:
            incomplete_dates.append(date_value)
    return complete_dates, incomplete_dates


def build_daily_stats(
    records_by_date: dict[str, list[dict[str, object]]],
    complete_dates: list[str],
) -> dict[str, dict[str, float]]:
    daily_stats: dict[str, dict[str, float]] = {}
    for date_value in complete_dates:
        records = first_48_periods(records_by_date[date_value])
        demands = [float(record["demand_mw"]) for record in records]
        usep_values = [float(record["usep_sgd_per_mwh"]) for record in records]
        daily_stats[date_value] = {
            "mean_demand": mean(demands),
            "max_demand": max(demands),
            "mean_usep": mean(usep_values),
            "max_usep": max(usep_values),
        }
    return daily_stats


def first_48_periods(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_period = {int(record["period"]): record for record in records}
    return [by_period[period] for period in range(1, REQUIRED_PERIODS + 1)]


def choose_selected_date(
    daily_stats: dict[str, dict[str, float]],
    target_date: str | None,
    selection: str,
) -> str:
    complete_dates = sorted(daily_stats.keys())
    if target_date:
        normalized_target = parse_date(target_date)
        if normalized_target not in daily_stats:
            raise ValueError(f"Target date {normalized_target} is not a complete NEMS day.")
        return normalized_target

    if selection == "first-complete":
        return complete_dates[0]
    if selection == "typical":
        all_day_mean = mean(stat["mean_demand"] for stat in daily_stats.values())
        return min(complete_dates, key=lambda date_value: abs(daily_stats[date_value]["mean_demand"] - all_day_mean))
    if selection == "peak-demand":
        return max(complete_dates, key=lambda date_value: daily_stats[date_value]["max_demand"])
    if selection == "peak-price":
        return max(complete_dates, key=lambda date_value: daily_stats[date_value]["max_usep"])
    if selection == "low-demand":
        return min(complete_dates, key=lambda date_value: daily_stats[date_value]["mean_demand"])
    raise ValueError(f"Unknown selection method: {selection}")


def aggregate_hourly_day(
    day_records: list[dict[str, object]],
    source_file: Path,
    scale_method: str,
    selection_method: str,
    global_mean_demand: float,
    hours: int = 24,
) -> list[dict[str, object]]:
    if hours <= 0 or hours > 24:
        raise ValueError(f"hours must be in [1, 24], got {hours}.")

    by_period = {int(record["period"]): record for record in day_records}
    hourly: list[dict[str, object]] = []
    has_solar = any("solar_mw" in record for record in day_records)

    for hour in range(hours):
        p1 = hour * 2 + 1
        p2 = p1 + 1
        if p1 not in by_period or p2 not in by_period:
            raise ValueError(f"Missing half-hour period pair {p1}-{p2}.")
        records = [by_period[p1], by_period[p2]]
        row: dict[str, object] = {
            "date": str(records[0]["date"]),
            "hour": hour,
            "period_start": p1,
            "period_end": p2,
            "demand_mw": mean(float(record["demand_mw"]) for record in records),
            "usep_sgd_per_mwh": mean(float(record["usep_sgd_per_mwh"]) for record in records),
            "source_file": str(source_file),
            "scale_method": scale_method,
            "selection_method": selection_method,
        }
        if has_solar:
            solar_values = [float(record.get("solar_mw", math.nan)) for record in records]
            finite_solar = [value for value in solar_values if math.isfinite(value)]
            row["solar_mw"] = mean(finite_solar) if finite_solar else math.nan
        hourly.append(row)

    day_demands = [float(row["demand_mw"]) for row in hourly]
    denominator = choose_scale_denominator(day_demands, scale_method, global_mean_demand)
    for row in hourly:
        row["grid_load_scale"] = float(row["demand_mw"]) / denominator
    return hourly


def choose_scale_denominator(day_demands: list[float], scale_method: str, global_mean_demand: float) -> float:
    if scale_method == "mean":
        denominator = mean(day_demands)
    elif scale_method == "max":
        denominator = max(day_demands)
    elif scale_method == "global-mean":
        denominator = global_mean_demand
    else:
        raise ValueError(f"Unknown scale method: {scale_method}")
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"Invalid demand scale denominator: {denominator}")
    return denominator


def build_all_days_hourly_profiles(
    records_by_date: dict[str, list[dict[str, object]]],
    complete_dates: list[str],
    source_file: Path,
    global_mean_demand: float,
    hours: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date_value in complete_dates:
        hourly = aggregate_hourly_day(
            day_records=records_by_date[date_value],
            source_file=source_file,
            scale_method="mean",
            selection_method="all-days",
            global_mean_demand=global_mean_demand,
            hours=hours,
        )
        day_mean_demand = mean(float(row["demand_mw"]) for row in hourly)
        for row in hourly:
            rows.append(
                {
                    "date": row["date"],
                    "hour": row["hour"],
                    "demand_mw": row["demand_mw"],
                    "usep_sgd_per_mwh": row["usep_sgd_per_mwh"],
                    "grid_load_scale_by_day_mean": float(row["demand_mw"]) / day_mean_demand,
                    "grid_load_scale_by_global_mean": float(row["demand_mw"]) / global_mean_demand,
                    "source_file": str(source_file),
                }
            )
    return rows


def write_output_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "date",
        "hour",
        "period_start",
        "period_end",
        "demand_mw",
        "usep_sgd_per_mwh",
    ]
    if any("solar_mw" in row for row in rows):
        fieldnames.append("solar_mw")
    fieldnames.extend([
        "grid_load_scale",
        "source_file",
        "scale_method",
        "selection_method",
    ])
    write_csv(path, fieldnames, rows)


def write_all_days_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "date",
        "hour",
        "demand_mw",
        "usep_sgd_per_mwh",
        "grid_load_scale_by_day_mean",
        "grid_load_scale_by_global_mean",
        "source_file",
    ]
    write_csv(path, fieldnames, rows)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
