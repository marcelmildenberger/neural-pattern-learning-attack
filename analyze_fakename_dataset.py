"""
Analyze fakename-style TSV datasets by computing simple bigram statistics.

Given a dataset (e.g., data/datasets/fakename_5k.tsv), this script produces
an analysis text file mirroring the existing *_analysis.txt outputs:
    - Average entry length (uncleaned, concatenated fields except uid)
    - Top-k character 2-grams after lowercasing and stripping non-alphanumerics
    - Macro-averaged precision, recall and F1 when always predicting that
      same top-k bigram set for every record.

If a second dataset (e.g., fakename_uniform.tsv) is provided via --uniform,
the script reuses the top-k bigrams from the main dataset as predictions for
the uniform dataset and reports precision/recall/F1 for that transfer setting.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence, Tuple


BIGRAM_RE = re.compile(r"[^a-z0-9]")


@dataclass
class AnalysisResult:
    average_entry_length: int
    top_bigrams: List[str]
    precision: float
    recall: float
    f1: float


def read_rows(tsv_path: Path) -> Tuple[List[dict], Sequence[str]]:
    with tsv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"{tsv_path} is missing a header row")
    return rows, reader.fieldnames


def entry_text(row: dict) -> str:
    """Concatenate all fields except uid to mirror earlier analyses."""
    return "".join(value for key, value in row.items() if key.lower() != "uid")


def clean(text: str) -> str:
    return BIGRAM_RE.sub("", text.lower())


def bigrams(text: str) -> List[str]:
    return [text[i : i + 2] for i in range(len(text) - 1)]


def precision_recall_f1(true_set: set, pred_set: set) -> Tuple[float, float, float]:
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def analyze_dataset(tsv_path: Path, top_k: int) -> AnalysisResult:
    rows, _ = read_rows(tsv_path)
    if not rows:
        raise ValueError(f"{tsv_path} is empty")

    lengths = []
    bigram_counter: Counter[str] = Counter()
    per_row_bigrams: List[set] = []

    for row in rows:
        raw = entry_text(row)
        lengths.append(len(raw))

        cleaned = clean(raw)
        grams_list = bigrams(cleaned)
        per_row_bigrams.append(set(grams_list))
        bigram_counter.update(grams_list)

    top_bigrams = [gram for gram, _ in bigram_counter.most_common(top_k)]
    pred_set = set(top_bigrams)

    precisions = []
    recalls = []
    f1s = []
    for grams in per_row_bigrams:
        p, r, f = precision_recall_f1(grams, pred_set)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return AnalysisResult(
        average_entry_length=round(mean(lengths)),
        top_bigrams=top_bigrams,
        precision=mean(precisions),
        recall=mean(recalls),
        f1=mean(f1s),
    )


def evaluate_on_target(tsv_path: Path, predicted_bigrams: Iterable[str]) -> Tuple[float, float, float]:
    rows, _ = read_rows(tsv_path)
    pred_set = set(predicted_bigrams)

    precisions = []
    recalls = []
    f1s = []

    for row in rows:
        cleaned = clean(entry_text(row))
        grams = set(bigrams(cleaned))
        p, r, f = precision_recall_f1(grams, pred_set)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return mean(precisions), mean(recalls), mean(f1s)


def write_analysis(out_path: Path, result: AnalysisResult, top_k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Average entry length: {result.average_entry_length}",
        f"Top {top_k} 2-grams: {result.top_bigrams}",
        f"Precision: {result.precision:.4f}",
        f"Recall: {result.recall:.4f}",
        f"F1 Score: {result.f1:.4f}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_uniform_eval(out_path: Path, source: Path, top_k: int, metrics: Tuple[float, float, float], top_bigrams: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, f1 = metrics
    lines = [
        f"Top {top_k} 2-grams source: {source.name}",
        f"Top {top_k} 2-grams: {top_bigrams}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1 Score: {f1:.4f}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bigram analysis for fakename datasets.")
    parser.add_argument("dataset", type=Path, help="Path to the main fakename TSV (e.g., fakename_5k.tsv).")
    parser.add_argument("--top-k", type=int, default=20, help="How many bigrams to keep (default: 20).")
    parser.add_argument("--output", type=Path, help="Where to write the main analysis (default: <dataset>_analysis.txt).")
    parser.add_argument(
        "--uniform",
        type=Path,
        help="Optional uniform TSV. The script will reuse the main dataset's top-k bigrams as predictions and score them here.",
    )
    parser.add_argument(
        "--uniform-output",
        type=Path,
        help="Output path for the uniform evaluation (default: <uniform>_baseline_from_<dataset_stem>_top<k>.txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    main_result = analyze_dataset(args.dataset, args.top_k)
    analysis_path = (
        args.output
        if args.output
        else args.dataset.with_name(f"{args.dataset.stem}_analysis.txt")
    )
    write_analysis(analysis_path, main_result, args.top_k)
    print(f"Wrote analysis to {analysis_path}")

    if args.uniform:
        uniform_out = (
            args.uniform_output
            if args.uniform_output
            else args.uniform.with_name(
                f"{args.uniform.stem}_baseline_from_{args.dataset.stem}_top{args.top_k}.txt"
            )
        )
        uniform_metrics = evaluate_on_target(args.uniform, main_result.top_bigrams)
        write_uniform_eval(uniform_out, args.dataset, args.top_k, uniform_metrics, main_result.top_bigrams)
        print(f"Wrote uniform baseline evaluation to {uniform_out}")


if __name__ == "__main__":
    main()
