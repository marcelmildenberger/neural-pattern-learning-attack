#!/usr/bin/env python3
"""
Prune duplicate experiment result folders by keeping the run with the best avg_dice
for each (encoding, dataset, overlap) combination.

By default the script only prints what it would delete. Use --delete to actually
remove directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_config(config_path: Path) -> Optional[dict]:
    """Load a JSON config file."""
    try:
        with config_path.open("r") as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"Skipping {config_path.parent.name}: config.json not found")
    except json.JSONDecodeError as exc:
        print(f"Skipping {config_path.parent.name}: invalid JSON ({exc})")
    except OSError as exc:
        print(f"Skipping {config_path.parent.name}: cannot read config ({exc})")
    return None


def read_avg_dice(metrics_path: Path) -> Optional[float]:
    """Return avg_dice from trained_model/metrics.csv, if present."""
    if not metrics_path.exists():
        return None

    try:
        with metrics_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                metric_name = (row.get("metric") or "").strip().lower()
                if metric_name == "avg_dice":
                    value = row.get("value")
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
    except OSError as exc:
        print(f"Could not read metrics at {metrics_path}: {exc}")
    return None


def normalize_overlap(raw_overlap) -> Optional[float]:
    """Normalize overlap so that small floating differences do not split keys."""
    if raw_overlap is None:
        return None
    try:
        return round(float(raw_overlap), 6)
    except (TypeError, ValueError):
        return None


def build_key(config: dict) -> Optional[Tuple[str, str, float]]:
    """Build the deduplication key (encoding, dataset, overlap)."""
    global_cfg = config.get("GLOBAL_CONFIG", {})
    enc_cfg = config.get("ENC_CONFIG", {})

    encoding = enc_cfg.get("AliceAlgo")
    data_path = global_cfg.get("Data")
    overlap = normalize_overlap(global_cfg.get("Overlap"))

    dataset = Path(data_path).name if data_path else None

    if encoding is None or dataset is None or overlap is None:
        return None
    return encoding, dataset, overlap


def collect_runs(base_dir: Path) -> Dict[Tuple[str, str, float], List[dict]]:
    """Collect experiment runs grouped by their deduplication key."""
    grouped: Dict[Tuple[str, str, float], List[dict]] = {}

    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        config = load_config(exp_dir / "config.json")
        if config is None:
            continue

        key = build_key(config)
        if key is None:
            print(f"Skipping {exp_dir.name}: missing encoding/dataset/overlap in config")
            continue

        avg_dice = read_avg_dice(exp_dir / "trained_model" / "metrics.csv")
        run_info = {
            "path": exp_dir,
            "avg_dice": avg_dice,
            "mtime": exp_dir.stat().st_mtime,
            "key": key,
        }
        grouped.setdefault(key, []).append(run_info)

    return grouped


def choose_best_and_deletions(
    grouped: Dict[Tuple[str, str, float], List[dict]]
) -> Tuple[Dict[Tuple[str, str, float], dict], List[dict]]:
    """Decide which runs to keep and which to delete."""
    keepers: Dict[Tuple[str, str, float], dict] = {}
    to_delete: List[dict] = []

    for key, runs in grouped.items():
        valid_runs = [r for r in runs if r["avg_dice"] is not None]
        if not valid_runs:
            print(
                f"No avg_dice found for combination {key}; not deleting any of "
                f"{len(runs)} run(s)."
            )
            continue

        def score(run: dict) -> Tuple[float, float]:
            dice = run["avg_dice"]
            return (dice if dice is not None else -math.inf, run["mtime"])

        best = max(valid_runs, key=score)
        keepers[key] = best

        for run in runs:
            if run is not best:
                to_delete.append(run)

    return keepers, to_delete


def delete_runs(runs: List[dict], base_dir: Path, execute: bool) -> None:
    """Delete or report deletion of the provided runs."""
    action = "Deleting" if execute else "Would delete"
    for run in runs:
        rel_path = run["path"].relative_to(base_dir.parent)
        dice = run["avg_dice"]
        dice_str = f"{dice:.6f}" if dice is not None else "missing"
        print(f"{action}: {rel_path} (avg_dice={dice_str})")
        if execute:
            shutil.rmtree(run["path"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate experiment results keeping best avg_dice."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiment_results"),
        help="Root directory that contains experiment folders.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete duplicate experiment folders (default is dry-run).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    if not base_dir.exists() or not base_dir.is_dir():
        raise SystemExit(f"{base_dir} does not exist or is not a directory")

    grouped = collect_runs(base_dir)
    keepers, to_delete = choose_best_and_deletions(grouped)

    total = sum(len(runs) for runs in grouped.values())
    print(f"Found {len(grouped)} unique combinations across {total} run(s).")
    print(f"Will keep {len(keepers)} best run(s).")

    if not to_delete:
        print("No duplicates to delete.")
        return

    delete_runs(to_delete, base_dir, execute=args.delete)
    if not args.delete:
        print("Dry run complete. Re-run with --delete to remove the directories.")


if __name__ == "__main__":
    main()
