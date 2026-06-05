import argparse
from pathlib import Path

import pandas as pd

from utils.encoder_registry import get_encoder_spec, supported_nepal_encoders


# === General Parameters ===
GLOBAL_CONFIG = {
    "Data": None,
    "Overlap": None,
    "DropFrom": None,
    "Verbose": False,
    "MatchingMetric": "cosine",
    "Matching": "MinWeight",
    "SaveAliceEncs": False,
    "SaveEveEncs": False,
    "DevMode": False,
    "BenchMode": True,
    "SaveResults": True,
    "UseGPU": True,
    "SaveModel": False,
    "SavePredictions": False,
    "UseNoisyDatasets": True,
    # If Graph Matching Attack is disabled, overlap will instead be used as the NEPAL training proportion.
    "GraphMatchingAttack": False,
}

# === NEPAL Training Parameters ===
NEPAL_CONFIG = {
    "ParallelTrials": 10,
    "TrainSize": 0.8,
    "Patience": 5,
    "MinDelta": 1e-4,
    "NumSamples": 175,
    "Epochs": 25,
    "MetricToOptimize": "average_dice",  # Options: "average_dice", "average_precision", ...
    "MatchingTechnique": "greedy",  # Options: "greedy"
    "EarlyStopThreshold": 0.99,
}

# === Encoding Parameters for Alice & Eve ===
ENC_CONFIG = {
    # Encoding technique
    "AliceAlgo": "",
    "AliceSecret": "SuperSecretSalt1337",
    "AliceN": 2,
    "AliceMetric": "dice",
    "EveAlgo": "",
    "EveSecret": "ATotallyDifferentString42",
    "EveN": 2,
    "EveMetric": "dice",

    # Bloom Filter specific
    "AliceBFLength": 1024,
    "AliceBits": 10,
    "AliceDiffuse": False,
    "AliceT": 10,
    "AliceEldLength": 1024,
    "EveBFLength": 1024,
    "EveBits": 10,
    "EveDiffuse": False,
    "EveT": 10,
    "EveEldLength": 1024,

    # Tabulation MinHash specific
    "AliceNHash": 1024,
    "AliceNHashBits": 64,
    "AliceNSubKeys": 8,
    "Alice1BitHash": True,
    "EveNHash": 1024,
    "EveNHashBits": 64,
    "EveNSubKeys": 8,
    "Eve1BitHash": True,

    # Two-Step Hashing specific
    "AliceNHashFunc": 10,
    "AliceNHashCol": 1000,
    "AliceRandMode": "PNG",
    "EveNHashFunc": 10,
    "EveNHashCol": 1000,
    "EveRandMode": "PNG",
}

# === Embedding Configuration (e.g., Node2Vec) ===
EMB_CONFIG = {
    "Algo": "Node2Vec",
    "AliceQuantile": 0.9,
    "AliceDiscretize": False,
    "AliceDim": 128,
    "AliceContext": 10,
    "AliceNegative": 1,
    "AliceNormalize": True,
    "EveQuantile": 0.9,
    "EveDiscretize": False,
    "EveDim": 128,
    "EveContext": 10,
    "EveNegative": 1,
    "EveNormalize": True,
    "AliceWalkLen": 100,
    "AliceNWalks": 20,
    "AliceP": 250,
    "AliceQ": 300,
    "AliceEpochs": 5,
    "AliceSeed": 42,
    "EveWalkLen": 100,
    "EveNWalks": 20,
    "EveP": 250,
    "EveQ": 300,
    "EveEpochs": 5,
    "EveSeed": 42,
}

# === Graph Alignment Config ===
ALIGN_CONFIG = {
    "RegWS": 0,
    "RegInit": 1,
    "Batchsize": 1,
    "LR": 200.0,
    "NIterWS": 100,
    "NIterInit": 5,
    "NEpochWS": 100,
    "LRDecay": 1,
    "Sqrt": True,
    "EarlyStopping": 10,
    "Selection": "None",
    "MaxLoad": None,
    "Wasserstein": True,
}

PAPER_DATASETS = [
    "fakename_1k.tsv",
    "fakename_2k.tsv",
    "fakename_5k.tsv",
    "fakename_10k.tsv",
    "fakename_20k.tsv",
    "fakename_50k.tsv",
]
PAPER_ENCODERS = ["BloomFilter", "TwoStepHash", "TabMinHash"]
PAPER_OVERLAPS = [0.2, 0.4, 0.6, 0.8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NEPAL experiment matrices. Defaults reproduce the paper-style synthetic noisy matrix."
    )
    parser.add_argument("--datasets", nargs="+", default=PAPER_DATASETS, help="Dataset file names or paths.")
    parser.add_argument("--encoders", nargs="+", default=PAPER_ENCODERS, help="Encoder names or aliases.")
    parser.add_argument("--overlaps", nargs="+", type=float, default=PAPER_OVERLAPS, help="Overlap/train proportions.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/datasets"), help="Directory for dataset files.")
    parser.add_argument("--clean", action="store_true", help="Use clean datasets instead of data/datasets/noisy.")
    parser.add_argument("--graph-matching", action="store_true", help="Enable GMA-NEPAL known-pair generation.")
    parser.add_argument(
        "--drop-from",
        nargs="+",
        choices=["Alice", "Eve", "Both"],
        default=None,
        help="Drop strategy for GMA mode. Defaults to Eve and Both when --graph-matching is set.",
    )
    parser.add_argument("--num-samples", type=int, default=NEPAL_CONFIG["NumSamples"], help="Ray Tune trials.")
    parser.add_argument("--epochs", type=int, default=NEPAL_CONFIG["Epochs"], help="Maximum training epochs.")
    parser.add_argument("--train-size", type=float, default=NEPAL_CONFIG["TrainSize"], help="Train share of known pairs.")
    parser.add_argument("--parallel-trials", type=int, default=NEPAL_CONFIG["ParallelTrials"], help="Parallel HPO trials.")
    parser.add_argument("--metric", default=NEPAL_CONFIG["MetricToOptimize"], help="HPO metric to optimize.")
    parser.add_argument("--early-stop-threshold", type=float, default=NEPAL_CONFIG["EarlyStopThreshold"])
    parser.add_argument("--bf-diffusion", action="store_true", help="Use *_bfd_encoded.tsv for BloomFilter runs.")
    parser.add_argument("--no-gpu", action="store_true", help="Disable CUDA usage.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose experiment output.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing them.")
    parser.add_argument("--max-runs", type=int, default=None, help="Execute at most this many planned runs.")
    parser.add_argument("--failed-output", default="experiment_results/failed_experiments.csv")
    return parser.parse_args()


def resolve_dataset_path(dataset: str, data_dir: Path, use_noisy: bool) -> str:
    path = Path(dataset)
    if path.parent != Path("."):
        return str(path)
    root = data_dir / "noisy" if use_noisy else data_dir
    return str(root / dataset)


def resolve_encoders(encoders: list[str]) -> list[str]:
    return [get_encoder_spec(encoder).name for encoder in encoders]


def planned_runs(args: argparse.Namespace):
    encoders = resolve_encoders(args.encoders)
    unsupported = sorted(set(encoders) - set(supported_nepal_encoders()))
    if unsupported:
        raise SystemExit(
            "These encoders do not have NEPAL dataset loaders yet: " + ", ".join(unsupported)
        )

    drop_from_values = args.drop_from
    if args.graph_matching and drop_from_values is None:
        drop_from_values = ["Eve", "Both"]
    if not args.graph_matching:
        drop_from_values = [""]

    use_noisy = not args.clean
    for dataset in args.datasets:
        data_path = resolve_dataset_path(dataset, args.data_dir, use_noisy)
        for encoding in encoders:
            for overlap in args.overlaps:
                for drop_from in drop_from_values:
                    yield {
                        "dataset": dataset,
                        "data_path": data_path,
                        "encoding": encoding,
                        "overlap": overlap,
                        "drop_from": drop_from,
                    }


def build_configs(run, args):
    global_config = GLOBAL_CONFIG.copy()
    enc_config = ENC_CONFIG.copy()
    emb_config = EMB_CONFIG.copy()
    align_config = ALIGN_CONFIG.copy()
    nepal_config = NEPAL_CONFIG.copy()

    global_config.update(
        {
            "Data": run["data_path"],
            "Overlap": run["overlap"],
            "DropFrom": run["drop_from"],
            "GraphMatchingAttack": args.graph_matching,
            "UseNoisyDatasets": not args.clean,
            "UseGPU": not args.no_gpu,
            "Verbose": args.verbose,
        }
    )
    nepal_config.update(
        {
            "NumSamples": args.num_samples,
            "Epochs": args.epochs,
            "TrainSize": args.train_size,
            "ParallelTrials": args.parallel_trials,
            "MetricToOptimize": args.metric,
            "EarlyStopThreshold": args.early_stop_threshold,
        }
    )
    enc_config["AliceAlgo"] = run["encoding"]
    enc_config["AliceDiffuse"] = args.bf_diffusion and run["encoding"] == "BloomFilter"
    enc_config["EveAlgo"] = "BloomFilter" if run["encoding"] == "BloomFilter" else "None"
    enc_config["EveDiffuse"] = enc_config["AliceDiffuse"] and enc_config["EveAlgo"] == "BloomFilter"
    return global_config, enc_config, emb_config, align_config, nepal_config


def main() -> int:
    args = parse_args()
    runs = list(planned_runs(args))
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    print(f"Planned NEPAL runs: {len(runs)}")
    for idx, run in enumerate(runs, start=1):
        label = f"{idx:04d}: {run['encoding']} {run['dataset']} overlap={run['overlap']} drop_from={run['drop_from'] or 'synthetic'}"
        if args.dry_run:
            print(label)

    if args.dry_run:
        return 0

    from nepal import run_nepal

    failed_experiments = []
    for idx, run in enumerate(runs, start=1):
        print(
            f"\n[{idx}/{len(runs)}] {run['encoding']} - {run['dataset']} - "
            f"{run['overlap']} - {run['drop_from'] or 'synthetic'}"
        )
        configs = build_configs(run, args)
        try:
            run_nepal(*configs)
        except Exception as e:
            failed_experiments.append(
                {
                    "encoding": run["encoding"],
                    "dataset": run["dataset"],
                    "data_path": run["data_path"],
                    "overlap": run["overlap"],
                    "drop_from": run["drop_from"],
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                }
            )
            print(f"Failed: {run['encoding']} - {run['dataset']} - {run['overlap']}: {e}")

    if failed_experiments:
        failed_path = Path(args.failed_output)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_df = pd.DataFrame(failed_experiments)
        failed_df.to_csv(failed_path, index=False)
        print(f"\nSaved {len(failed_experiments)} failed experiments to {failed_path}")
    else:
        print("\nNo failed experiments to save.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
