import copy
import json
import os
import time
from datetime import datetime
from functools import partial
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import warnings
from utils.hyperparameter_training import hyperparameter_training
from utils.early_stopping import EarlyStopping
from graphMatching.gma import run_gma
from utils.pytorch_base_model import BaseModel
from utils.utils import *
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def run_nepal(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, NEPAL_CONFIG):
    """
    Main experiment entry point for running NEPAL
    """
    # Ignore optuna warnings.
    warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
    # Set default values for alignment and global configuration.
    # ALIGN_CONFIG["RegWS"] is set to the maximum of 0.1 and one third of the overlap parameter.
    # GLOBAL_CONFIG["Workers"] is set to the number of available CPU cores
    ALIGN_CONFIG["RegWS"] = max(0.1, GLOBAL_CONFIG["Overlap"] / 3)
    GLOBAL_CONFIG["Workers"] = max_cpu_cores = os.cpu_count()
    
    # Validate and set parallel trials configuration
    parallel_trials = NEPAL_CONFIG["ParallelTrials"]
    if parallel_trials > max_cpu_cores:
        print(f"[WARNING] ParallelTrials ({parallel_trials}) exceeds available CPU cores ({max_cpu_cores}). Setting to {max_cpu_cores}.")
        parallel_trials = max_cpu_cores
        NEPAL_CONFIG["ParallelTrials"] = parallel_trials
    
    # Validate and set GPU configuration
    use_gpu = GLOBAL_CONFIG["UseGPU"]
    gpu_count = 0
    
    if use_gpu:
        # Auto-detect and use maximum available GPUs when UseGPU is True
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count == 0:
            print("[WARNING] UseGPU is True but no GPUs available. Disabling GPU usage.")
            GLOBAL_CONFIG["UseGPU"] = use_gpu = False
        else:
            print(f"[INFO] Using {gpu_count} available GPU(s)")

    

    # Create a unique experiment directory for saving results and configuration.
    # The directory name encodes the algorithm, dataset, and timestamp for traceability.
    # All configuration dictionaries are saved to a config.txt file in this directory for reproducibility.
    selected_dataset = GLOBAL_CONFIG["Data"].split("/")[-1].replace(".tsv", "")
    experiment_tag = "experiment_" + ENC_CONFIG["AliceAlgo"] + "_" + selected_dataset + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_experiment_directory = f"experiment_results/{experiment_tag}"
    os.makedirs(current_experiment_directory, exist_ok=True)

    all_configs = {
        "GLOBAL_CONFIG": GLOBAL_CONFIG,
        "NEPAL_CONFIG": NEPAL_CONFIG,
        "ENC_CONFIG": ENC_CONFIG,
        "EMB_CONFIG": EMB_CONFIG,
        "ALIGN_CONFIG": ALIGN_CONFIG
    }
    
    with open(os.path.join(current_experiment_directory, "config.json"), "w") as f:
        json.dump(all_configs, f, indent=4)
    
    # Initialize elapsed times to 0 (will be updated if timing is enabled)
    elapsed_gma = 0
    elapsed_hyperparameter_optimization = 0
    elapsed_model_training = 0
    elapsed_application_to_encoded_data = 0
    elapsed_refinement_and_reconstruction = 0
    elapsed_total = 0
    
    # Current NEPAL implementation only supports Two-Gram encoding but can be extended in the future.
    # Generate all possible two-character combinations (2-grams) from lowercase letters and digits.
    # This includes letter-letter, letter-digit, and digit-digit pairs.
    # The resulting list `all_bi_grams` is used for encoding/decoding tasks,
    # and `bi_gram_dict` maps each index to its corresponding 2-gram.
    all_bi_grams = get_all_bi_grams()
    bi_gram_dict = {i: bi_gram for i, bi_gram in enumerate(all_bi_grams)}


    # Start timing the total run and the GMA run.
    if GLOBAL_CONFIG["BenchMode"]:
        start_total = time.time()
    if GLOBAL_CONFIG["BenchMode"] and GLOBAL_CONFIG["GraphMatchingAttack"]:
        start_gma = time.time()

    # Get the hashes for the encoding and embedding.
    eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(
        GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG
    )

    # Get the data directory and identifier for the current run to check if the data is already available. 
    gma_enabled = GLOBAL_CONFIG["GraphMatchingAttack"]
    data_dir = os.path.abspath("./data")
    suffix = "" if gma_enabled else "_synthetic"
    noisy = "_noisy" if GLOBAL_CONFIG["UseNoisyDatasets"] else ""
    alice_enc_hash = alice_enc_hash + suffix + noisy
    identifier = f"{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}"
    path_reidentified = f"{data_dir}/available_to_eve/reidentified_individuals_{identifier}.h5"
    path_not_reidentified = f"{data_dir}/available_to_eve/not_reidentified_individuals_{identifier}.h5"
    path_all = f"{data_dir}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5"


    # Check if Data is Available or Needs to be Generated
    if not(os.path.isfile(path_reidentified) and os.path.isfile(path_not_reidentified) and os.path.isfile(path_all)):
        if GLOBAL_CONFIG["GraphMatchingAttack"]:
            run_gma(
                GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG,
                eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash
            )
        # When GraphMatchingAttack is disabled, create synthetic data splits based on overlap
        else:
            create_synthetic_data_splits(
                GLOBAL_CONFIG, ENC_CONFIG, data_dir, alice_enc_hash, identifier, 
                path_reidentified, path_not_reidentified, path_all
            )

    if GLOBAL_CONFIG["BenchMode"] and GLOBAL_CONFIG["GraphMatchingAttack"] and start_gma is not None:
        elapsed_gma = time.time() - start_gma

    # Load the experiment datasets (train, val, test) and check for empty splits.
    # If any split is empty, write a termination log and exit early to prevent downstream errors.
    datasets = load_experiment_datasets(data_dir, alice_enc_hash, identifier, ENC_CONFIG, NEPAL_CONFIG, GLOBAL_CONFIG, all_bi_grams, splits=("train", "val", "test"))
    data_train, data_val, data_test = datasets["train"], datasets["val"], datasets["test"]
    if len(data_train) == 0 or len(data_val) == 0 or len(data_test) == 0:
        # Save termination log as CSV for better analysis and exit early to avoid crashes downstream
        termination_data = {
            "metric": ["Status", "Length of data_train", "Length of data_val", "Length of data_test"],
            "value": ["Training process canceled due to empty dataset", len(data_train), len(data_val), len(data_test)]
        }
        termination_df = pd.DataFrame(termination_data)
        termination_csv_path = os.path.join(current_experiment_directory, "termination_log.csv")
        termination_df.to_csv(termination_csv_path, index=False)

        if GLOBAL_CONFIG["BenchMode"] and start_total is not None:
            elapsed_total = time.time() - start_total
            save_nepal_runtime_log(
                elapsed_gma=elapsed_gma,
                elapsed_hyperparameter_optimization=elapsed_hyperparameter_optimization,
                elapsed_model_training=elapsed_model_training,
                elapsed_application_to_encoded_data=elapsed_application_to_encoded_data,
                elapsed_refinement_and_reconstruction=elapsed_refinement_and_reconstruction,
                elapsed_total=elapsed_total,
                output_dir=current_experiment_directory
            )

        if GLOBAL_CONFIG.get("Verbose", False):
            print("[INFO] Terminating run because one or more dataset splits are empty.")
        return 1
    # Start timing the hyperparameter optimization run.
    if GLOBAL_CONFIG["BenchMode"]:
        start_hyperparameter_optimization = time.time()

    search_space = {
        "output_dim": len(all_bi_grams),
        "num_layers": tune.randint(1, 3),
        "hidden_layer": tune.choice([512, 1024, 2048, 4096]),
        "dropout_rate": tune.uniform(0.1, 0.4),
        "activation_fn": tune.choice(["elu", "selu", "tanh"]),
        "optimizer": tune.choice([
            {"name": "Adam", "lr": tune.loguniform(1e-5, 1e-3)},
            {"name": "AdamW", "lr": tune.loguniform(1e-5, 1e-3)},
            {"name": "RMSprop", "lr": tune.loguniform(1e-5, 1e-3)},
        ]),
        "loss_fn": tune.choice(["BCEWithLogitsLoss", "MultiLabelSoftMarginLoss", "SoftMarginLoss"]),
        "threshold": tune.uniform(0.2, 0.7),
        "lr_scheduler": tune.choice([
            {"name": "ReduceLROnPlateau", "mode": "min", "factor": tune.uniform(0.1, 0.5), "patience": tune.choice([5, 10, 15])},
            {"name": "CosineAnnealingLR", "T_max": tune.loguniform(10, 50), "eta_min": tune.choice([1e-5, 1e-6, 0])},
            {"name": "CyclicLR", "base_lr": tune.loguniform(1e-5, 1e-3), "max_lr": tune.loguniform(1e-3, 1e-1), "step_size_up": tune.choice([2000, 4000]), "mode_cyclic": tune.choice(["triangular", "triangular2", "exp_range"]) },
            {"name": "None"}
        ]),
        "batch_size": tune.choice([8, 16, 32, 64]),
    }

    # Initialize Ray for hyperparameter optimization.
    ray.init(
        num_cpus=GLOBAL_CONFIG["Workers"],
        num_gpus=gpu_count if use_gpu else 0,
        ignore_reinit_error=True,
        logging_level="ERROR"
    )

    # Initialize the Optuna search and the ASHAScheduler.
    optuna_search = OptunaSearch(metric=NEPAL_CONFIG["MetricToOptimize"], mode="max")
    scheduler = ASHAScheduler(metric="total_val_loss", mode="min")

    # Define the trainable function.
    trainable = partial(
        hyperparameter_training,
        data_dir=data_dir,
        output_dim=len(all_bi_grams),
        alice_enc_hash=alice_enc_hash,
        identifier=identifier,
        patience=NEPAL_CONFIG["Patience"],
        min_delta=NEPAL_CONFIG["MinDelta"],
        workers=GLOBAL_CONFIG["Workers"] // 15 if GLOBAL_CONFIG["UseGPU"] else 0,
        ENC_CONFIG=ENC_CONFIG,
        NEPAL_CONFIG=NEPAL_CONFIG,
        GLOBAL_CONFIG=GLOBAL_CONFIG,
        bi_gram_dict=bi_gram_dict,
        all_bi_grams=all_bi_grams
    )

    # Calculate resources per trial based on parallel trials and available resources
    cpu_per_trial = max(1, GLOBAL_CONFIG["Workers"] // parallel_trials)
    gpu_per_trial = (gpu_count / parallel_trials) if use_gpu and gpu_count > 0 else 0
    
    # Wrap the trainable function with resources
    trainable_with_resources = tune.with_resources(
        trainable,
        resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial}
    )

    # Add early stopping based on optimization metric threshold (default 0.99, configurable)
    early_stop_threshold = NEPAL_CONFIG.get("EarlyStopThreshold", 0.99)

    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=NEPAL_CONFIG["NumSamples"],
            # Stop a trial as soon as the optimization metric reaches the
            # desired threshold (e.g., Dice >= 0.99).
            stop={NEPAL_CONFIG["MetricToOptimize"]: early_stop_threshold},
        ),
        param_space=search_space,
        run_config=train.RunConfig(
            name="nepal_hpo",
        ),
    )

    # Run the hyperparameter optimization.
    results = tuner.fit()

    # Shut down Ray.
    ray.shutdown()

    # Stop timing the hyperparameter optimization.
    if GLOBAL_CONFIG["BenchMode"] and start_hyperparameter_optimization is not None:
        elapsed_hyperparameter_optimization = time.time() - start_hyperparameter_optimization

    # Save the results of hyperparameter optimization to a CSV file and a plot.
    hyperparameter_optimization_directory = f"{current_experiment_directory}/hyperparameteroptimization"
    os.makedirs(hyperparameter_optimization_directory, exist_ok=True)

    # Get the best result.
    best_result = results.get_best_result(metric=NEPAL_CONFIG["MetricToOptimize"], mode="max")
    if GLOBAL_CONFIG["SaveResults"]:
        print_and_save_result("best_result", best_result, hyperparameter_optimization_directory)

    best_config = resolve_config(best_result.config)

    # Start timing the model training.
    if GLOBAL_CONFIG["BenchMode"]:
        start_model_training = time.time()

    datasets = load_experiment_datasets(data_dir, alice_enc_hash, identifier, ENC_CONFIG, NEPAL_CONFIG, GLOBAL_CONFIG, all_bi_grams, splits=("train", "val", "test"))
    data_train, data_val, data_test = datasets["train"], datasets["val"], datasets["test"]

    input_dim=data_train[0][0].shape[0]
    dataloader_train = DataLoader(
        data_train,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=True,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 15 if GLOBAL_CONFIG["UseGPU"] else 0,
    )
    dataloader_val = DataLoader(
        data_val,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=False,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 15 if GLOBAL_CONFIG["UseGPU"] else 0,
    )
    dataloader_test = DataLoader(
        data_test,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=False,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 15 if GLOBAL_CONFIG["UseGPU"] else 0,
    )

    # Respect the user's GPU toggle; fall back to CPU even if CUDA is available when UseGPU is False
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    
    model = BaseModel(
        input_dim=input_dim,
        output_dim=len(all_bi_grams),
        hidden_layer=best_config.get("hidden_layer", 128),
        num_layers=best_config.get("num_layers", 2),
        dropout_rate=best_config.get("dropout_rate", 0.2),
        activation_fn=best_config.get("activation_fn", "relu")
    )
    
    model.to(device)

    # Initialize the TensorBoard writer.
    if GLOBAL_CONFIG["SaveResults"]:
        run_name = "".join([
            best_config.get("loss_fn", "MultiLabelSoftMarginLoss"),
            best_config.get("optimizer").get("name", "Adam"),
            ENC_CONFIG["AliceAlgo"],
            best_config.get("activation_fn", "relu"),
        ])
        tb_writer = SummaryWriter(f"{current_experiment_directory}/{run_name}")

    # Initialize the loss function.
    match best_config.get("loss_fn", "MultiLabelSoftMarginLoss"):
        case "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
        case "MultiLabelSoftMarginLoss":
            criterion = nn.MultiLabelSoftMarginLoss(reduction='mean')
        case "SoftMarginLoss":
            criterion = nn.SoftMarginLoss()
        case _:
            raise ValueError(f"Unsupported loss function: {best_config.get('loss_fn', 'MultiLabelSoftMarginLoss')}")

    # Initialize the optimizer.
    match best_config.get("optimizer").get("name", "Adam"):
        case "Adam":
            optimizer = optim.Adam(model.parameters(), lr=best_config.get("optimizer").get("lr"))
        case "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=best_config.get("optimizer").get("lr"))
        case "SGD":
            optimizer = optim.SGD(model.parameters(),
                                lr=best_config.get("optimizer").get("lr"),
                                momentum=best_config.get("optimizer").get("momentum"))
        case "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=best_config.get("optimizer").get("lr"))
        case _:
            raise ValueError(f"Unsupported optimizer: {best_config.get('optimizer').get('name', 'Adam')}")

                
    match best_config.get("lr_scheduler").get("name", "None"):
        case "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=best_config.get("lr_scheduler").get("step_size"),
                gamma=best_config.get("lr_scheduler").get("gamma")
            )
        case "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=best_config.get("lr_scheduler").get("gamma")
            )
        case "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=best_config.get("lr_scheduler").get("mode"),
                factor=best_config.get("lr_scheduler").get("factor"),
                patience=best_config.get("lr_scheduler").get("patience")
            )
        case "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=best_config.get("lr_scheduler").get("T_max")
            )
        case "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=best_config.get("lr_scheduler").get("base_lr"),
                max_lr=best_config.get("lr_scheduler").get("max_lr"),
                step_size_up=best_config.get("lr_scheduler").get("step_size_up"),
                mode=best_config.get("lr_scheduler").get("mode_cyclic"),
                cycle_momentum=False
            )
        case None | "None":
            scheduler = None
        case _:
            raise ValueError(f"Unsupported LR scheduler: {best_config.get('lr_scheduler').get('name', 'None')}")

    def train_final_model(model, dataloader_train, dataloader_val, criterion, optimizer, device, scheduler=None):
        num_epochs = best_config.get("epochs", NEPAL_CONFIG["Epochs"])
        verbose = GLOBAL_CONFIG["Verbose"]
        patience = NEPAL_CONFIG["Patience"]
        min_delta = NEPAL_CONFIG["MinDelta"]
        best_val_loss = float('inf')
        best_model_state = None
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, verbose=verbose)
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            train_loss = run_epoch(
                model, dataloader_train, criterion, optimizer,
                device, is_training=True, verbose=verbose, scheduler=scheduler
            )
            train_losses.append(train_loss)
            val_loss = run_epoch(
                model, dataloader_val, criterion, optimizer,
                device, is_training=False, verbose=verbose, scheduler=scheduler
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
            val_losses.append(val_loss)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()

            log_epoch_metrics(epoch, num_epochs, train_loss, val_loss, tb_writer=tb_writer if 'tb_writer' in locals() else None, save_results=GLOBAL_CONFIG["SaveResults"])
            if early_stopper(val_loss):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model, train_losses, val_losses

    # Train the final model.
    model, train_losses, val_losses = train_final_model(
        model, dataloader_train, dataloader_val,
        criterion, optimizer, device,
        scheduler=scheduler
    )

    # Stop timing the model training.
    if GLOBAL_CONFIG["BenchMode"] and start_model_training is not None:
        elapsed_model_training = time.time() - start_model_training

    # Define the paths for the model, config, result, and metrics files.
    trained_model_directory = os.path.join(
        current_experiment_directory,
        "trained_model"
    )
    os.makedirs(trained_model_directory, exist_ok=True)

    # Plot the training and validation loss curves.
    plot_loss_curves(
        train_losses,
        val_losses,
        save_path=f"{trained_model_directory}/loss_curve.png",
        save=GLOBAL_CONFIG["SaveResults"]
    )

    # Save the trained model and config if requested.
    if GLOBAL_CONFIG["SaveModel"]:
        torch.save(model.state_dict(), os.path.join(trained_model_directory, "model.pt"))
        with open(os.path.join(trained_model_directory, "config.json"), "w") as f:
            json.dump(best_config, f, indent=4)

    # Start timing the application to encoded data.
    if GLOBAL_CONFIG["BenchMode"]:
        start_application_to_encoded_data = time.time()

    # Initialize the metrics.
    total_dice = total_precision = total_recall = total_f1 = 0.0
    num_samples = 0

    # Initialize the results.
    results = []

    # Initialize the threshold.
    threshold = best_config.get("threshold", 0.5)

    model.eval()
    dataloader_iter = tqdm(dataloader_test, desc="Test loop") if GLOBAL_CONFIG["Verbose"] else dataloader_test
    # Prepare per-bigram accounting structures
    bi_gram_to_idx = {v: k for k, v in bi_gram_dict.items()}
    num_bi_grams = len(all_bi_grams)
    pred_counts = [0] * num_bi_grams  # how often each bi-gram was predicted
    true_counts = [0] * num_bi_grams  # how often each bi-gram appears in ground truth
    tp_counts = [0] * num_bi_grams    # true positives per bi-gram

    with torch.no_grad():
        for data, labels, uids in dataloader_iter:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            actual_bi_grams = decode_labels_to_bi_grams(bi_gram_dict, labels)
            predicted_scores = map_probabilities_to_bi_grams(bi_gram_dict, probs)
            predicted_filtered = filter_high_scoring_bi_grams(predicted_scores, threshold)
            bs = data.size(0)
            dice, precision, recall, f1 = calculate_performance_metrics(actual_bi_grams, predicted_filtered)
            total_dice += dice
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_samples += bs

            # Update per-bigram counts (use sets to avoid double-counting within a record)
            for actual_set, predicted_set in zip(map(set, actual_bi_grams), map(set, predicted_filtered)):
                for gram in actual_set:
                    idx = bi_gram_to_idx.get(gram)
                    if idx is not None:
                        true_counts[idx] += 1
                for gram in predicted_set:
                    idx = bi_gram_to_idx.get(gram)
                    if idx is not None:
                        pred_counts[idx] += 1
                for gram in actual_set & predicted_set:
                    idx = bi_gram_to_idx.get(gram)
                    if idx is not None:
                        tp_counts[idx] += 1

            for uid, actual, predicted in zip(uids, actual_bi_grams, predicted_filtered):
                metrics = metrics_per_entry(actual, predicted)
                results.append({
                    "uid": uid,
                    "actual_bi_grams": actual,
                    "predicted_bi_grams": predicted,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "dice": metrics["dice"],
                    "jaccard": metrics["jaccard"]
                })
    avg_dice = total_dice / num_samples
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_f1 = total_f1 / num_samples

    # --- Per-bigram precision / recall / F1 and dataset frequency ---
    df_all_records = load_dataframe(path_all)
    dataset_occurrence_counts = [0] * num_bi_grams  # number of records containing each bi-gram
    for _, row in df_all_records.iterrows():
        # Drop encoding + uid (last two columns) to mirror training label construction
        record_string = "".join(row.iloc[:-2].astype(str))
        grams_in_record = set(extract_bi_grams(record_string))
        for gram in grams_in_record:
            idx = bi_gram_to_idx.get(gram)
            if idx is not None:
                dataset_occurrence_counts[idx] += 1
    dataset_total_records = len(df_all_records)

    per_bigram_rows = []
    for gram in all_bi_grams:
        idx = bi_gram_to_idx[gram]
        tp = tp_counts[idx]
        pred = pred_counts[idx]
        true = true_counts[idx]
        precision_bg = tp / pred if pred else 0.0
        recall_bg = tp / true if true else 0.0
        f1_bg = 2 * precision_bg * recall_bg / (precision_bg + recall_bg) if (precision_bg + recall_bg) else 0.0
        dataset_pct = dataset_occurrence_counts[idx] / dataset_total_records if dataset_total_records else 0.0
        per_bigram_rows.append({
            "bigram": gram,
            "precision": precision_bg,
            "recall": recall_bg,
            "f1": f1_bg,
            "tp": tp,
            "pred_count": pred,
            "true_count": true,
            "dataset_pct": dataset_pct
        })

    per_bigram_df = pd.DataFrame(per_bigram_rows)
    per_bigram_df.to_csv(os.path.join(current_experiment_directory, "per_bigram_metrics.csv"), index=False)

    # Stop timing the application to encoded data.
    if GLOBAL_CONFIG["BenchMode"] and start_application_to_encoded_data is not None:
        elapsed_application_to_encoded_data = time.time() - start_application_to_encoded_data

    # Save the metrics and results if requested.
    if GLOBAL_CONFIG["SaveResults"]:
        # Save metrics as CSV for better analysis
        metrics_data = {
            "metric": ["avg_precision", "avg_recall", "avg_f1", "avg_dice"],
            "value": [avg_precision, avg_recall, avg_f1, avg_dice]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f"{trained_model_directory}/metrics.csv", index=False)

    if GLOBAL_CONFIG["SavePredictions"]:
        with open(f"{trained_model_directory}/results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    # Normalize the results.
    results_df = pd.json_normalize(results)

    # Plot the metric distributions.
    plot_metric_distributions(
        results_df,
        trained_model_directory,
        save=GLOBAL_CONFIG["SaveResults"]
    )

    # Start timing the refinement and reconstruction.
    if GLOBAL_CONFIG["BenchMode"]:
        start_refinement_and_reconstruction = time.time()

    header = read_header(GLOBAL_CONFIG["Data"])
    
    df_not_reid_cached = get_not_reidentified_df(data_dir, identifier, alice_enc_hash=alice_enc_hash)

    run_reidentification_greedy(results, header, df_not_reid_cached, current_experiment_directory=current_experiment_directory)

    # Stop timing the refinement and reconstruction.
    if GLOBAL_CONFIG["BenchMode"] and start_total is not None:
        if start_refinement_and_reconstruction is not None:
            elapsed_refinement_and_reconstruction = time.time() - start_refinement_and_reconstruction
        elapsed_total = time.time() - start_total
        save_nepal_runtime_log(
            elapsed_gma=elapsed_gma,
            elapsed_hyperparameter_optimization=elapsed_hyperparameter_optimization,
            elapsed_model_training=elapsed_model_training,
            elapsed_application_to_encoded_data=elapsed_application_to_encoded_data,
            elapsed_refinement_and_reconstruction=elapsed_refinement_and_reconstruction,
            elapsed_total=elapsed_total,
            output_dir=current_experiment_directory
        )

    return 0
