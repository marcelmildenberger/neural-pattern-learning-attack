from nepal import run_nepal
import pandas as pd

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
    "UseNoisyDatasets": True, #Not Tested For GMA
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
    "EarlyStopThreshold": 0.99
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

# List to store failed experiments
failed_experiments = []

encs = ["BloomFilter", "TwoStepHash", "TabMinHash"]

datasets = [
#    "titanic_full.tsv",  
    "fakename_1k.tsv",
    "fakename_2k.tsv",
    "fakename_5k.tsv",
    "fakename_10k.tsv",
    "fakename_20k.tsv",
#    "euro_person.tsv", 
    "fakename_50k.tsv", 
]

dataset_overlaps = [0.2, 0.4, 0.6, 0.8]

drop_from = ["Eve", "Both"] if GLOBAL_CONFIG["GraphMatchingAttack"] else [""]

datasets_path = "./data/datasets/"
datasets_path += "noisy/" if GLOBAL_CONFIG["UseNoisyDatasets"] else ""

for dataset in datasets:
    for encoding in encs:
        for ov in dataset_overlaps:
            for df in drop_from:
                GLOBAL_CONFIG["DropFrom"] = df
                ENC_CONFIG["AliceAlgo"] = encoding
                ENC_CONFIG["EveAlgo"] = "None"
                if encoding == "BloomFilter":
                    ENC_CONFIG["EveAlgo"] = encoding
                GLOBAL_CONFIG["Data"] = f"{datasets_path}{dataset}"
                GLOBAL_CONFIG["Overlap"] = ov
                try:
                    run_nepal(
                        GLOBAL_CONFIG.copy(),
                        ENC_CONFIG.copy(),
                        EMB_CONFIG.copy(),
                        ALIGN_CONFIG.copy(),
                        NEPAL_CONFIG.copy()
                    )
                except Exception as e:
                    # Record failed experiment
                    failed_experiments.append({
                        "encoding": encoding,
                        "dataset": dataset,
                        "overlap": ov,
                        "error_message": str(e),
                        "drop_from": df,
                        "error_type": type(e).__name__,
                    })
                    print(f"Failed: {encoding} - {dataset} - {ov}: {str(e)}")
                    continue


# Save failed experiments to CSV
if failed_experiments:
    failed_df = pd.DataFrame(failed_experiments)
    failed_df.to_csv("experiment_results/failed_experiments.csv", index=False)
    print(f"\nSaved {len(failed_experiments)} failed experiments to failed_experiments.csv")
else:
    print("\nNo failed experiments to save.")