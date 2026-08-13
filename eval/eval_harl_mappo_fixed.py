"""Formal fixed-suite deterministic evaluation entry for MAPPO/HAPPO actors."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "harl_mappo_fixed_eval.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate dual-agent MAPPO/HAPPO actors on a fixed scenario suite.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite-id")
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--generate-suite", action="store_true")
    parser.add_argument("--overwrite-suite", action="store_true")
    parser.add_argument("--generate-suite-only", action="store_true")
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--model-dir", action="append", type=Path, default=[])
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-id")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--harl-source", type=Path, default=Path(os.environ.get("HARL_SOURCE_PATH", PROJECT_ROOT.parent / "HARL")))
    parser.add_argument("--runtime-path", type=Path, default=Path(os.environ["HARL_RUNTIME_PATH"]) if os.environ.get("HARL_RUNTIME_PATH") else None)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    from train.train_harl_mappo_short import configure_import_paths, load_yaml_config

    harl_source = configure_import_paths(args.harl_source, args.runtime_path)
    config = load_yaml_config(args.config)
    required = {"main", "environment", "evaluation", "suites"}
    if missing := sorted(required.difference(config)):
        raise ValueError(f"Fixed evaluation config is missing sections: {missing}.")
    suite_id = args.suite_id or str(config["evaluation"]["suite_id"])
    if suite_id not in config["suites"]:
        raise ValueError(f"Unknown configured suite {suite_id!r}.")
    suite_config = config["suites"][suite_id]
    if config["evaluation"].get("deterministic") is not True:
        raise ValueError("Formal evaluation config must set deterministic=true.")
    if int(config["evaluation"].get("n_eval_workers", 0)) != 1:
        raise ValueError("Formal fixed evaluation requires n_eval_workers=1.")
    if int(config["environment"].get("episode_length", 0)) != 24:
        raise ValueError("Formal fixed evaluation requires episode_length=24.")

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import torch

    torch.set_num_threads(int(config["evaluation"].get("torch_threads", 1)))
    from marl.evaluation.evaluator import FixedPolicyEvaluator
    from marl.evaluation.fixed_scenario_suite import generate_suite, load_suite
    from marl.evaluation.model_loader import load_model

    suite_dir = (args.suite_dir or PROJECT_ROOT / config["evaluation"]["suite_root"] / suite_id).resolve()
    if args.generate_suite:
        generate_suite(
            suite_id=suite_id,
            suite_type=str(suite_config["suite_type"]),
            scenario_seeds=list(suite_config["scenario_seeds"]),
            output_dir=suite_dir,
            experiment_case=str(config["environment"]["experiment_case"]),
            harl_source=harl_source,
            overwrite=args.overwrite_suite,
        )
    suite = load_suite(suite_dir)
    if args.generate_suite_only:
        result = {"status": "suite_generated", "suite_dir": str(suite_dir), "suite_sha256": suite["suite_sha256"]}
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return result

    sources = [("run_dir", value) for value in args.run_dir] + [("checkpoint", value) for value in args.checkpoint] + [("model_dir", value) for value in args.model_dir]
    if not sources:
        raise ValueError("At least one --run-dir, --checkpoint, or --model-dir is required.")
    models = []
    for index, (kind, value) in enumerate(sources):
        kwargs = {kind: value, "model_config": args.model_config, "device": args.device}
        model = load_model(**kwargs)
        if any(existing.model_id == model.model_id for existing in models):
            model.model_id = f"{model.model_id}-{index}"
        models.append(model)

    evaluation_id = args.evaluation_id or datetime.now(timezone.utc).strftime("eval-%Y%m%d-%H%M%S")
    model_group = models[0].model_id if len(models) == 1 else f"comparison-{len(models)}-models"
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / config["evaluation"]["output_root"] / suite_id / model_group / evaluation_id
    ).resolve()
    evaluator = FixedPolicyEvaluator(device=args.device, deterministic=True)
    result = evaluator.evaluate(
        suite_dir=suite_dir,
        models=models,
        output_dir=output_dir,
        evaluation_id=evaluation_id,
        scenario_ids=args.scenario_id or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
