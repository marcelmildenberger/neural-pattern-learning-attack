import os
import json
import pandas as pd
from pathlib import Path

def read_json_file(file_path):
    """Read and parse JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def read_csv_file(file_path):
    """Read and parse CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_reidentification_rate(summary_df):
    """Extract reidentification rate from summary CSV"""
    if summary_df is None:
        return None
    
    for _, row in summary_df.iterrows():
        metric_raw = str(row.get('metric', '')).strip().lower().replace('_', ' ')
        if 'reidentification' in metric_raw and 'rate' in metric_raw:
            value = row.get('value', '')
            is_percent = isinstance(value, str) and '%' in value
            if isinstance(value, str):
                value = value.strip().replace('%', '')
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
            if is_percent or rate > 1:
                rate /= 100
            return rate
    return None

def extract_metrics(metrics_df):
    """Extract metrics from trained model CSV"""
    if metrics_df is None:
        return {}
    
    metrics = {}
    for _, row in metrics_df.iterrows():
        metric_name = row['metric']
        value = row['value']
        # Convert numeric values
        try:
            if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                value = float(value)
        except:
            pass
        metrics[metric_name] = value
    return metrics

def extract_reidentification_info(results_dir, matching_technique=None):
    """Extract re-identification summary matching the configured technique, if available."""
    info = {
        "ReidentificationMethod": None,
        "ReidentificationRate": None,
        "TotalReidentifiedIndividuals": None,
        "TotalNotReidentifiedIndividuals": None,
    }
    normalized_matching = str(matching_technique or "").strip().lower()
    if normalized_matching in {"", "skip", "none", "dice_only"}:
        return info
    if not results_dir.exists():
        return info

    summary_files = sorted(results_dir.glob("summary_*.csv"))
    if not summary_files:
        return info

    target_file = None
    if normalized_matching:
        candidate = results_dir / f"summary_{normalized_matching}.csv"
        if candidate.exists():
            target_file = candidate
    if target_file is None:
        target_file = summary_files[0]

    summary_df = read_csv_file(target_file)
    if summary_df is None:
        return info

    metric_map = {}
    for _, row in summary_df.iterrows():
        metric = str(row.get("metric", "")).strip().lower()
        metric_map[metric] = row.get("value")

    rate = extract_reidentification_rate(summary_df)
    method = metric_map.get("reidentification_method")
    total_reidentified = metric_map.get("total_reidentified_individuals")
    total_not_reidentified = metric_map.get("total_not_reidentified_individuals")

    def _to_int(value):
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                return int(value)
        return None

    info["ReidentificationMethod"] = method if isinstance(method, str) and method else target_file.stem.replace("summary_", "")
    info["ReidentificationRate"] = rate
    info["TotalReidentifiedIndividuals"] = _to_int(total_reidentified)
    info["TotalNotReidentifiedIndividuals"] = _to_int(total_not_reidentified)
    return info

def extract_runtime(runtime_df):
    """Extract runtime information from NEPAL runtime log"""
    if runtime_df is None:
        return {}
    
    runtimes = {}
    for _, row in runtime_df.iterrows():
        phase = row['phase']
        # Convert phase names to match expected format
        if phase == "gma":
            key = "GraphMatchingAttack"
        elif phase == "hyperparameter_optimization":
            key = "HyperparameterOptimization"
        elif phase == "model_training":
            key = "ModelTraining"
        elif phase == "application_to_encoded_data":
            key = "ApplicationtoEncodedData"
        elif phase == "refinement_and_reconstruction":
            key = "RefinementandReconstruction"
        elif phase == "total_runtime":
            key = "TotalRuntime"
        else:
            key = phase.replace(" ", "")
        
        runtimes[key] = row['runtime_minutes']
    return runtimes

def extract_experiment_data(exp_dir):
    """Extract all data from a single experiment directory"""
    exp_path = Path(exp_dir)
    
    # Check for termination log - skip if present
    if (exp_path / "termination_log.csv").exists():
        return None
    
    # Read configuration
    config = read_json_file(exp_path / "config.json")
    if not config:
        return None
    
    global_config = config.get("GLOBAL_CONFIG", {})
    enc_config = config.get("ENC_CONFIG", {})
    nepal_config = config.get("NEPAL_CONFIG", {})

    # Read metrics
    metrics_df = read_csv_file(exp_path / "trained_model" / "metrics.csv")
    metrics = extract_metrics(metrics_df)
    if not metrics:
        return None
    
    # Read re-identification results based on configured matching technique
    matching_technique = nepal_config.get("MatchingTechnique") or global_config.get("MatchingTechnique")
    reid_info = extract_reidentification_info(exp_path / "re_identification_results", matching_technique)
    
    # Read runtime data (may not exist if BenchMode was disabled)
    runtime_df = read_csv_file(exp_path / "nepal_runtime_log.csv")
    runtime = extract_runtime(runtime_df)
    
    # Read hyperparameter optimization results (only exists if HPO was enabled)
    best_result = None
    if (exp_path / "hyperparameteroptimization" / "best_result.json").exists():
        best_result = read_json_file(exp_path / "hyperparameteroptimization" / "best_result.json")
    
    # Create record with basic information
    record = {
        "ExperimentFolder": exp_path.name,
        "Encoding": enc_config.get("AliceAlgo"),
        "Dataset": os.path.basename(global_config.get("Data", "")),
        "DropFrom": global_config.get("DropFrom"),
        "Overlap": global_config.get("Overlap"),
        "GraphMatchingAttack": global_config.get("GraphMatchingAttack", True),
        "MatchingTechnique": matching_technique,
        "MatchingMetric": global_config.get("MatchingMetric"),
        "TrainedPrecision": metrics.get("avg_precision"),
        "TrainedRecall": metrics.get("avg_recall"),
        "TrainedF1": metrics.get("avg_f1"),
        "TrainedDice": metrics.get("avg_dice"),
    }
    record.update(reid_info)
    
    # Add hyperparameter optimization results (only if HPO was enabled)
    if best_result:
        record.update({
            "HypOpOutputDim": best_result.get("output_dim"),
            "HypOpNumLayers": best_result.get("num_layers"),
            "HypOpHiddenSize": best_result.get("hidden_layer"),
            "HypOpDropout": best_result.get("dropout_rate"),
            "HypOpActivation": best_result.get("activation_fn"),
            "HypOpOptimizer": best_result.get("optimizer"),
            "HypOpLossFn": best_result.get("loss_fn"),
            "HypOpThreshold": best_result.get("threshold"),
            "HypOpLRScheduler": best_result.get("lr_scheduler"),
            "HypOpBatchSize": best_result.get("batch_size"),
            "HypOpValLoss": best_result.get("total_val_loss"),
            "HypOpEpochs": best_result.get("epochs"),
            "HypOpF1": best_result.get("average_f1"),
            "HypOpPrecision": best_result.get("average_precision"),
            "HypOpRecall": best_result.get("average_recall"),
            "HypOpDice": best_result.get("average_dice"),
            "LenTrain": best_result.get("len_train"),
            "LenVal": best_result.get("len_val"),
        })
    
    # Add runtime information (only if BenchMode was enabled)
    if runtime:
        record.update({
            "GraphMatchingAttackTime": runtime.get("GraphMatchingAttack"),
            "HyperparameterOptimizationTime": runtime.get("HyperparameterOptimization"),
            "ModelTrainingTime": runtime.get("ModelTraining"),
            "ApplicationtoEncodedDataTime": runtime.get("ApplicationtoEncodedData"),
            "RefinementandReconstructionTime": runtime.get("RefinementandReconstruction"),
            "TotalRuntime": runtime.get("TotalRuntime"),
        })
    else:
        # Add empty runtime fields when BenchMode was disabled
        runtime_fields = [
            "GraphMatchingAttackTime", "HyperparameterOptimizationTime", "ModelTrainingTime",
            "ApplicationtoEncodedDataTime", "RefinementandReconstructionTime", "TotalRuntime"
        ]
        for field in runtime_fields:
            record[field] = None
    
    return record

def main():
    """Main function to extract all experiment results"""
    base_path = "experiment_results"
    output_file = "formatted_results.csv"
    
    # Find all experiment directories
    experiment_dirs = list(Path(base_path).glob("experiment_*"))
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Extract data from each experiment
    data_records = []
    skipped_count = 0
    missing_config_count = 0
    missing_metrics_count = 0
    
    # Track experiment configurations
    gma_enabled_count = 0
    gma_disabled_count = 0
    
    for exp_dir in experiment_dirs:
        print(f"Processing {exp_dir.name}...")
        
        # Check for termination log
        if (exp_dir / "termination_log.csv").exists():
            skipped_count += 1
            print(f"  Skipped (terminated): {exp_dir.name}")
            continue
        
        # Extract data
        record = extract_experiment_data(exp_dir)
        
        if record is None:
            # Check what's missing
            if not (exp_dir / "config.json").exists():
                missing_config_count += 1
                print(f"  Missing config: {exp_dir.name}")
            elif not (exp_dir / "trained_model" / "metrics.csv").exists():
                missing_metrics_count += 1
                print(f"  Missing metrics: {exp_dir.name}")
            else:
                print(f"  Failed to extract: {exp_dir.name}")
            continue
        
        # Track configuration statistics
        if record.get("GraphMatchingAttack", True):
            gma_enabled_count += 1
        else:
            gma_disabled_count += 1
        
        data_records.append(record)
        print(f"  Successfully extracted: {exp_dir.name}")
    
    # Create DataFrame and save
    df = pd.DataFrame(data_records)
    
    print(f"\nExtraction Summary:")
    print(f"Total experiments found: {len(experiment_dirs)}")
    print(f"Experiments with termination logs: {skipped_count}")
    print(f"Experiments missing config: {missing_config_count}")
    print(f"Experiments missing metrics: {missing_metrics_count}")
    print(f"Successfully extracted: {len(data_records)}")
    
    if len(data_records) > 0:
        print(f"\nExperiment Configuration Summary:")
        print(f"GraphMatchingAttack enabled: {gma_enabled_count}")
        print(f"GraphMatchingAttack disabled: {gma_disabled_count}")
        
        # Show encoding algorithm distribution
        if "Encoding" in df.columns:
            encoding_counts = df["Encoding"].value_counts()
            print(f"\nEncoding Algorithm Distribution:")
            for encoding, count in encoding_counts.items():
                print(f"  {encoding}: {count}")
        
        # Show dataset distribution
        if "Dataset" in df.columns:
            dataset_counts = df["Dataset"].value_counts()
            print(f"\nDataset Distribution:")
            for dataset, count in dataset_counts.items():
                print(f"  {dataset}: {count}")
        
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    else:
        print("No valid experiments found to extract")

if __name__ == "__main__":
    main()
