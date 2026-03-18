import copy
import json
import logging
import os
from datetime import datetime
from functools import partial

import pandas as pd
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from graphMatching.gma import run_gma
from pytorch_datasets.roundbased_encoding_scheme_dataset import RoundBasedEncodingSchemeDataset
from utils.early_stopping import EarlyStopping
from utils.hyperparameter_training import hyperparameter_training
from utils.modeling import (
    calculate_performance_metrics,
    decode_labels_to_bi_grams,
    filter_high_scoring_bi_grams,
    get_hashes,
    log_epoch_metrics,
    map_probabilities_to_bi_grams,
    metrics_per_entry,
    plot_loss_curves,
    plot_metric_distributions,
    print_and_save_result,
    resolve_config,
    run_epoch,
    save_nepal_runtime_log,
)
from utils.pytorch_base_model import BaseModel
from utils.data_pipeline import (
    create_synthetic_data_splits,
    export_gma_results,
    get_not_reidentified_df,
    load_dataframe,
    load_experiment_datasets,
    read_header,
    run_reidentification_greedy,
)
from utils.string_utils import extract_bi_grams, get_all_bi_grams


def _dataloader_workers(GLOBAL_CONFIG):
    return GLOBAL_CONFIG["Workers"] // 15 if GLOBAL_CONFIG["UseGPU"] else 0


def prepare_run_context(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, NEPAL_CONFIG, logger):
    ALIGN_CONFIG["RegWS"] = max(0.1, GLOBAL_CONFIG["Overlap"] / 3)
    GLOBAL_CONFIG["Workers"] = max_cpu_cores = os.cpu_count()

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Starting NEPAL run | dataset=%s | alice_algo=%s | graph_matching=%s | bench_mode=%s | use_gpu=%s",
            GLOBAL_CONFIG.get("Data"),
            ENC_CONFIG.get("AliceAlgo"),
            GLOBAL_CONFIG.get("GraphMatchingAttack"),
            GLOBAL_CONFIG.get("BenchMode"),
            GLOBAL_CONFIG.get("UseGPU"),
        )
        logger.info(
            "Derived params: RegWS=%.3f | Workers=%s | ParallelTrials=%s",
            ALIGN_CONFIG["RegWS"],
            GLOBAL_CONFIG["Workers"],
            NEPAL_CONFIG.get("ParallelTrials"),
        )

    unsupported_gma_encodings = {"Saul", "RoundBasedEncoder"}
    if GLOBAL_CONFIG["GraphMatchingAttack"] and (
        ENC_CONFIG.get("AliceAlgo") in unsupported_gma_encodings
        or ENC_CONFIG.get("EveAlgo") in unsupported_gma_encodings
    ):
        raise ValueError(
            "This encoding is currently supported only with synthetic splits. Set 'GraphMatchingAttack' to false."
        )

    parallel_trials = NEPAL_CONFIG["ParallelTrials"]
    if parallel_trials > max_cpu_cores:
        logger.warning(
            "ParallelTrials (%s) exceeds available CPU cores (%s). Setting to %s.",
            parallel_trials,
            max_cpu_cores,
            max_cpu_cores,
        )
        parallel_trials = max_cpu_cores
        NEPAL_CONFIG["ParallelTrials"] = parallel_trials

    use_gpu = GLOBAL_CONFIG["UseGPU"]
    gpu_count = 0
    if use_gpu:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count == 0:
            logger.warning("UseGPU is True but no GPUs available. Disabling GPU usage.")
            GLOBAL_CONFIG["UseGPU"] = use_gpu = False
        else:
            logger.info("Using %s available GPU(s)", gpu_count)

    selected_dataset = GLOBAL_CONFIG["Data"].split("/")[-1].replace(".tsv", "")
    experiment_tag = "experiment_" + ENC_CONFIG["AliceAlgo"] + "_" + selected_dataset + "_" + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    current_experiment_directory = f"experiment_results/{experiment_tag}"
    os.makedirs(current_experiment_directory, exist_ok=True)
    if logger.isEnabledFor(logging.INFO):
        logger.info("Results directory: %s", current_experiment_directory)

    all_configs = {
        "GLOBAL_CONFIG": GLOBAL_CONFIG,
        "NEPAL_CONFIG": NEPAL_CONFIG,
        "ENC_CONFIG": ENC_CONFIG,
        "EMB_CONFIG": EMB_CONFIG,
        "ALIGN_CONFIG": ALIGN_CONFIG,
    }
    with open(os.path.join(current_experiment_directory, "config.json"), "w") as f:
        json.dump(all_configs, f, indent=4)

    all_bi_grams = get_all_bi_grams()
    bi_gram_dict = {i: bi_gram for i, bi_gram in enumerate(all_bi_grams)}

    eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG)
    gma_enabled = GLOBAL_CONFIG["GraphMatchingAttack"]
    data_dir = os.path.abspath("./data")
    suffix = "" if gma_enabled else "_synthetic"
    noisy = "_noisy" if GLOBAL_CONFIG["UseNoisyDatasets"] else ""
    alice_enc_hash = alice_enc_hash + suffix + noisy
    identifier = f"{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}"

    return {
        "all_configs": all_configs,
        "all_bi_grams": all_bi_grams,
        "bi_gram_dict": bi_gram_dict,
        "current_experiment_directory": current_experiment_directory,
        "data_dir": data_dir,
        "eve_enc_hash": eve_enc_hash,
        "alice_enc_hash": alice_enc_hash,
        "eve_emb_hash": eve_emb_hash,
        "alice_emb_hash": alice_emb_hash,
        "identifier": identifier,
        "path_reidentified": f"{data_dir}/available_to_eve/reidentified_individuals_{identifier}.h5",
        "path_not_reidentified": f"{data_dir}/available_to_eve/not_reidentified_individuals_{identifier}.h5",
        "path_all": f"{data_dir}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5",
        "parallel_trials": parallel_trials,
        "use_gpu": use_gpu,
        "gpu_count": gpu_count,
    }


def ensure_input_artifacts(context, GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, logger):
    cached_paths = (
        context["path_reidentified"],
        context["path_not_reidentified"],
        context["path_all"],
    )
    if not all(os.path.isfile(path) for path in cached_paths):
        if logger.isEnabledFor(logging.INFO):
            logger.info("Derived identifier %s not cached; generating data artifacts.", context["identifier"])
        if GLOBAL_CONFIG["GraphMatchingAttack"]:
            if logger.isEnabledFor(logging.INFO):
                logger.info("Running Graph Matching Attack preprocessing.")
            run_gma(
                GLOBAL_CONFIG,
                ENC_CONFIG,
                EMB_CONFIG,
                ALIGN_CONFIG,
                context["eve_enc_hash"],
                context["alice_enc_hash"],
                context["eve_emb_hash"],
                context["alice_emb_hash"],
            )
        else:
            if logger.isEnabledFor(logging.INFO):
                logger.info("GraphMatchingAttack disabled; creating synthetic data splits.")
            create_synthetic_data_splits(
                GLOBAL_CONFIG,
                ENC_CONFIG,
                context["data_dir"],
                context["alice_enc_hash"],
                context["identifier"],
                context["path_reidentified"],
                context["path_not_reidentified"],
                context["path_all"],
            )
    else:
        if logger.isEnabledFor(logging.INFO):
            logger.info("Reusing cached data artifacts for identifier %s", context["identifier"])

    if GLOBAL_CONFIG["GraphMatchingAttack"]:
        export_gma_results(
            data_dir=context["data_dir"],
            identifier=context["identifier"],
            alice_enc_hash=context["alice_enc_hash"],
            eve_enc_hash=context["eve_enc_hash"],
            GLOBAL_CONFIG=GLOBAL_CONFIG,
            output_dir=context["current_experiment_directory"],
        )
        if logger.isEnabledFor(logging.INFO):
            logger.info("Saved GMA artifacts to %s/gma_results", context["current_experiment_directory"])


def load_datasets_or_terminate(context, ENC_CONFIG, NEPAL_CONFIG, GLOBAL_CONFIG, logger, timings):
    datasets = load_experiment_datasets(
        context["data_dir"],
        context["alice_enc_hash"],
        context["identifier"],
        ENC_CONFIG,
        NEPAL_CONFIG,
        GLOBAL_CONFIG,
        context["all_bi_grams"],
        splits=("train", "val", "test"),
    )
    data_train, data_val, data_test = datasets["train"], datasets["val"], datasets["test"]
    if len(data_train) == 0 or len(data_val) == 0 or len(data_test) == 0:
        termination_df = pd.DataFrame(
            {
                "metric": ["Status", "Length of data_train", "Length of data_val", "Length of data_test"],
                "value": [
                    "Training process canceled due to empty dataset",
                    len(data_train),
                    len(data_val),
                    len(data_test),
                ],
            }
        )
        termination_df.to_csv(
            os.path.join(context["current_experiment_directory"], "termination_log.csv"),
            index=False,
        )

        if GLOBAL_CONFIG["BenchMode"] and timings.get("elapsed_total") is not None:
            save_nepal_runtime_log(
                elapsed_gma=timings["elapsed_gma"],
                elapsed_hyperparameter_optimization=timings["elapsed_hyperparameter_optimization"],
                elapsed_model_training=timings["elapsed_model_training"],
                elapsed_application_to_encoded_data=timings["elapsed_application_to_encoded_data"],
                elapsed_refinement_and_reconstruction=timings["elapsed_refinement_and_reconstruction"],
                elapsed_total=timings["elapsed_total"],
                output_dir=context["current_experiment_directory"],
            )

        if logger.isEnabledFor(logging.INFO):
            logger.info("Terminating run because one or more dataset splits are empty.")
        return None

    return datasets


def _build_search_space(output_dim):
    return {
        "output_dim": output_dim,
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
            {
                "name": "ReduceLROnPlateau",
                "mode": "min",
                "factor": tune.uniform(0.1, 0.5),
                "patience": tune.choice([5, 10, 15]),
            },
            {"name": "CosineAnnealingLR", "T_max": tune.loguniform(10, 50), "eta_min": tune.choice([1e-5, 1e-6, 0])},
            {
                "name": "CyclicLR",
                "base_lr": tune.loguniform(1e-5, 1e-3),
                "max_lr": tune.loguniform(1e-3, 1e-1),
                "step_size_up": tune.choice([2000, 4000]),
                "mode_cyclic": tune.choice(["triangular", "triangular2", "exp_range"]),
            },
            {"name": "None"},
        ]),
        "batch_size": tune.choice([8, 16, 32, 64]),
    }


def run_hyperparameter_search(context, GLOBAL_CONFIG, ENC_CONFIG, NEPAL_CONFIG, logger):
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Starting hyperparameter search: samples=%s | metric=%s | early_stop=%.2f",
            NEPAL_CONFIG["NumSamples"],
            NEPAL_CONFIG["MetricToOptimize"],
            NEPAL_CONFIG.get("EarlyStopThreshold", 0.99),
        )

    ray.init(
        num_cpus=GLOBAL_CONFIG["Workers"],
        num_gpus=context["gpu_count"] if context["use_gpu"] else 0,
        ignore_reinit_error=True,
        logging_level="ERROR",
    )

    trainable = partial(
        hyperparameter_training,
        data_dir=context["data_dir"],
        output_dim=len(context["all_bi_grams"]),
        alice_enc_hash=context["alice_enc_hash"],
        identifier=context["identifier"],
        patience=NEPAL_CONFIG["Patience"],
        min_delta=NEPAL_CONFIG["MinDelta"],
        workers=_dataloader_workers(GLOBAL_CONFIG),
        ENC_CONFIG=ENC_CONFIG,
        NEPAL_CONFIG=NEPAL_CONFIG,
        GLOBAL_CONFIG=GLOBAL_CONFIG,
        bi_gram_dict=context["bi_gram_dict"],
        all_bi_grams=context["all_bi_grams"],
    )

    cpu_per_trial = max(1, GLOBAL_CONFIG["Workers"] // context["parallel_trials"])
    gpu_per_trial = (
        context["gpu_count"] / context["parallel_trials"]
        if context["use_gpu"] and context["gpu_count"] > 0
        else 0
    )
    trainable_with_resources = tune.with_resources(
        trainable,
        resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(metric=NEPAL_CONFIG["MetricToOptimize"], mode="max"),
            scheduler=ASHAScheduler(metric="total_val_loss", mode="min"),
            num_samples=NEPAL_CONFIG["NumSamples"],
        ),
        run_config=RunConfig(name="nepal_hpo", verbose=1),
        param_space=_build_search_space(len(context["all_bi_grams"])),
    )

    results = tuner.fit()
    ray.shutdown()

    hyperparameter_optimization_directory = f"{context['current_experiment_directory']}/hyperparameteroptimization"
    os.makedirs(hyperparameter_optimization_directory, exist_ok=True)

    best_result = results.get_best_result(metric=NEPAL_CONFIG["MetricToOptimize"], mode="max")
    if GLOBAL_CONFIG["SaveResults"]:
        print_and_save_result("best_result", best_result, hyperparameter_optimization_directory)

    best_metric_value = None
    try:
        best_metric_value = best_result.metrics.get(NEPAL_CONFIG["MetricToOptimize"])
    except Exception:
        best_metric_value = None
    if logger.isEnabledFor(logging.INFO):
        logger.info("Best trial %s = %s", NEPAL_CONFIG["MetricToOptimize"], best_metric_value)

    return best_result, resolve_config(best_result.config)


def prepare_training_run(context, GLOBAL_CONFIG, ENC_CONFIG, NEPAL_CONFIG, best_config, logger):
    datasets = load_experiment_datasets(
        context["data_dir"],
        context["alice_enc_hash"],
        context["identifier"],
        ENC_CONFIG,
        NEPAL_CONFIG,
        GLOBAL_CONFIG,
        context["all_bi_grams"],
        splits=("train", "val", "test"),
    )
    data_train, data_val, data_test = datasets["train"], datasets["val"], datasets["test"]

    batch_size = int(best_config.get("batch_size", 32))
    dataloader_train = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=_dataloader_workers(GLOBAL_CONFIG),
    )
    dataloader_val = DataLoader(
        data_val,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=_dataloader_workers(GLOBAL_CONFIG),
    )
    dataloader_test = DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=_dataloader_workers(GLOBAL_CONFIG),
    )

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Data prepared | train=%d | val=%d | test=%d | batch_size=%d",
            len(data_train),
            len(data_val),
            len(data_test),
            batch_size,
        )

    analysis_data_path = GLOBAL_CONFIG.get("AnalysisData")
    if analysis_data_path:
        if ENC_CONFIG["AliceAlgo"] not in {"Saul", "RSE", "RoundBasedEncoder"}:
            raise ValueError("AnalysisData override currently supports encoded_vector-style datasets only (Saul/RSE).")
        analysis_df = pd.read_csv(analysis_data_path, sep="\t")
        analysis_dataset = RoundBasedEncodingSchemeDataset(
            analysis_df,
            is_labeled=True,
            all_bi_grams=context["all_bi_grams"],
            dev_mode=GLOBAL_CONFIG["DevMode"],
        )
        dataloader_test = DataLoader(
            analysis_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=_dataloader_workers(GLOBAL_CONFIG),
        )

    input_dim = data_train[0][0].shape[0]
    device = torch.device("cuda:0" if context["use_gpu"] and torch.cuda.is_available() else "cpu")
    model = BaseModel(
        input_dim=input_dim,
        output_dim=len(context["all_bi_grams"]),
        hidden_layer=best_config.get("hidden_layer", 128),
        num_layers=best_config.get("num_layers", 2),
        dropout_rate=best_config.get("dropout_rate", 0.2),
        activation_fn=best_config.get("activation_fn", "relu"),
    )
    model.to(device)

    tb_writer = None
    if GLOBAL_CONFIG["SaveResults"]:
        optimizer_cfg = best_config.get("optimizer", {})
        run_name = "".join(
            [
                best_config.get("loss_fn", "MultiLabelSoftMarginLoss"),
                optimizer_cfg.get("name", "Adam"),
                ENC_CONFIG["AliceAlgo"],
                best_config.get("activation_fn", "relu"),
            ]
        )
        tb_writer = SummaryWriter(f"{context['current_experiment_directory']}/{run_name}")

    criterion = _create_loss(best_config)
    optimizer = _create_optimizer(model, best_config)
    scheduler = _create_scheduler(optimizer, best_config)

    trained_model_directory = os.path.join(context["current_experiment_directory"], "trained_model")
    os.makedirs(trained_model_directory, exist_ok=True)

    return {
        "datasets": datasets,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "dataloader_test": dataloader_test,
        "device": device,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "tb_writer": tb_writer,
        "trained_model_directory": trained_model_directory,
    }


def _create_loss(best_config):
    loss_name = best_config.get("loss_fn", "MultiLabelSoftMarginLoss")
    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(reduction="mean")
    if loss_name == "MultiLabelSoftMarginLoss":
        return nn.MultiLabelSoftMarginLoss(reduction="mean")
    if loss_name == "SoftMarginLoss":
        return nn.SoftMarginLoss()
    raise ValueError(f"Unsupported loss function: {loss_name}")


def _create_optimizer(model, best_config):
    optimizer_cfg = best_config.get("optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "Adam")
    lr = optimizer_cfg.get("lr")
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr)
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=optimizer_cfg.get("momentum"))
    if optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _create_scheduler(optimizer, best_config):
    scheduler_cfg = best_config.get("lr_scheduler", {})
    scheduler_name = scheduler_cfg.get("name", "None")
    if scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size"),
            gamma=scheduler_cfg.get("gamma"),
        )
    if scheduler_name == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_cfg.get("gamma"))
    if scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode"),
            factor=scheduler_cfg.get("factor"),
            patience=scheduler_cfg.get("patience"),
        )
    if scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_cfg.get("T_max"))
    if scheduler_name == "CyclicLR":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=scheduler_cfg.get("base_lr"),
            max_lr=scheduler_cfg.get("max_lr"),
            step_size_up=scheduler_cfg.get("step_size_up"),
            mode=scheduler_cfg.get("mode_cyclic"),
            cycle_momentum=False,
        )
    if scheduler_name in {None, "None"}:
        return None
    raise ValueError(f"Unsupported LR scheduler: {scheduler_name}")


def train_final_model(model_bundle, best_config, NEPAL_CONFIG, GLOBAL_CONFIG, logger):
    model = model_bundle["model"]
    criterion = model_bundle["criterion"]
    optimizer = model_bundle["optimizer"]
    scheduler = model_bundle["scheduler"]
    num_epochs = best_config.get("epochs", NEPAL_CONFIG["Epochs"])
    verbose = GLOBAL_CONFIG["Verbose"]
    early_stopper = EarlyStopping(
        patience=NEPAL_CONFIG["Patience"],
        min_delta=NEPAL_CONFIG["MinDelta"],
        verbose=verbose,
    )
    best_val_loss = float("inf")
    best_model_state = None
    train_losses = []
    val_losses = []

    if logger.isEnabledFor(logging.INFO):
        logger.info("Training final model for up to %s epochs", num_epochs)

    for epoch in range(num_epochs):
        train_loss = run_epoch(
            model,
            model_bundle["dataloader_train"],
            criterion,
            optimizer,
            model_bundle["device"],
            is_training=True,
            verbose=verbose,
            scheduler=scheduler,
        )
        val_loss = run_epoch(
            model,
            model_bundle["dataloader_val"],
            criterion,
            optimizer,
            model_bundle["device"],
            is_training=False,
            verbose=verbose,
            scheduler=scheduler,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        log_epoch_metrics(
            epoch,
            num_epochs,
            train_loss,
            val_loss,
            tb_writer=model_bundle["tb_writer"],
            save_results=GLOBAL_CONFIG["SaveResults"],
        )
        if early_stopper(val_loss):
            if logger.isEnabledFor(logging.INFO):
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def save_training_artifacts(context, GLOBAL_CONFIG, best_config, model_bundle, train_losses, val_losses):
    plot_loss_curves(
        train_losses,
        val_losses,
        save_path=f"{model_bundle['trained_model_directory']}/loss_curve.png",
        save=GLOBAL_CONFIG["SaveResults"],
    )

    if GLOBAL_CONFIG["SaveModel"]:
        torch.save(model_bundle["model"].state_dict(), os.path.join(model_bundle["trained_model_directory"], "model.pt"))
        with open(os.path.join(model_bundle["trained_model_directory"], "config.json"), "w") as f:
            json.dump(best_config, f, indent=4)


def evaluate_model(model_bundle, context, GLOBAL_CONFIG, best_config, logger):
    total_dice = total_precision = total_recall = total_f1 = 0.0
    rand_total_dice = rand_total_precision = rand_total_recall = rand_total_f1 = 0.0
    num_samples = 0
    results = []

    threshold = best_config.get("threshold", 0.5)
    bi_gram_to_idx = {v: k for k, v in context["bi_gram_dict"].items()}
    num_bi_grams = len(context["all_bi_grams"])

    df_all_records = load_dataframe(context["path_all"])
    dataset_occurrence_counts = [0] * num_bi_grams
    for _, row in df_all_records.iterrows():
        record_string = "".join(row.iloc[:-2].astype(str))
        grams_in_record = set(extract_bi_grams(record_string))
        for gram in grams_in_record:
            idx = bi_gram_to_idx.get(gram)
            if idx is not None:
                dataset_occurrence_counts[idx] += 1
    dataset_total_records = len(df_all_records)
    dataset_probabilities = [
        count / dataset_total_records if dataset_total_records else 0.0
        for count in dataset_occurrence_counts
    ]
    random_predict_probs = torch.tensor(dataset_probabilities, device=model_bundle["device"])

    pred_counts = [0] * num_bi_grams
    true_counts = [0] * num_bi_grams
    tp_counts = [0] * num_bi_grams
    rand_pred_counts = [0] * num_bi_grams
    rand_tp_counts = [0] * num_bi_grams

    model_bundle["model"].eval()
    with torch.no_grad():
        for data, labels, uids in model_bundle["dataloader_test"]:
            data, labels = data.to(model_bundle["device"]), labels.to(model_bundle["device"])
            logits = model_bundle["model"](data)
            probs = torch.sigmoid(logits)
            actual_bi_grams = decode_labels_to_bi_grams(context["bi_gram_dict"], labels)
            predicted_scores = map_probabilities_to_bi_grams(context["bi_gram_dict"], probs)
            predicted_filtered = filter_high_scoring_bi_grams(predicted_scores, threshold)

            rand_mask = torch.rand((data.size(0), num_bi_grams), device=model_bundle["device"]) < random_predict_probs
            rand_mask_cpu = rand_mask.cpu()
            random_predicted_filtered = []
            for rand_row in rand_mask_cpu:
                rand_indices = torch.nonzero(rand_row, as_tuple=False).view(-1).tolist()
                random_predicted_filtered.append([context["bi_gram_dict"][idx] for idx in rand_indices])

            bs = data.size(0)
            dice, precision, recall, f1 = calculate_performance_metrics(actual_bi_grams, predicted_filtered)
            rand_dice, rand_precision, rand_recall, rand_f1 = calculate_performance_metrics(
                actual_bi_grams,
                random_predicted_filtered,
            )

            total_dice += dice
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            rand_total_dice += rand_dice
            rand_total_precision += rand_precision
            rand_total_recall += rand_recall
            rand_total_f1 += rand_f1

            num_samples += bs

            for actual_set, predicted_set, rand_pred_set in zip(
                map(set, actual_bi_grams),
                map(set, predicted_filtered),
                map(set, random_predicted_filtered),
            ):
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
                for gram in rand_pred_set:
                    idx = bi_gram_to_idx.get(gram)
                    if idx is not None:
                        rand_pred_counts[idx] += 1
                for gram in actual_set & rand_pred_set:
                    idx = bi_gram_to_idx.get(gram)
                    if idx is not None:
                        rand_tp_counts[idx] += 1

            for uid, actual, predicted in zip(uids, actual_bi_grams, predicted_filtered):
                metrics = metrics_per_entry(actual, predicted)
                results.append(
                    {
                        "uid": uid,
                        "actual_bi_grams": actual,
                        "predicted_bi_grams": predicted,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "dice": metrics["dice"],
                        "jaccard": metrics["jaccard"],
                    }
                )

    avg_metrics = {
        "avg_dice": total_dice / num_samples,
        "avg_precision": total_precision / num_samples,
        "avg_recall": total_recall / num_samples,
        "avg_f1": total_f1 / num_samples,
        "rand_avg_dice": rand_total_dice / num_samples,
        "rand_avg_precision": rand_total_precision / num_samples,
        "rand_avg_recall": rand_total_recall / num_samples,
        "rand_avg_f1": rand_total_f1 / num_samples,
    }

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Evaluation complete | avg_dice=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
            avg_metrics["avg_dice"],
            avg_metrics["avg_precision"],
            avg_metrics["avg_recall"],
            avg_metrics["avg_f1"],
        )

    per_bigram_rows = []
    for gram in context["all_bi_grams"]:
        idx = bi_gram_to_idx[gram]
        tp = tp_counts[idx]
        pred = pred_counts[idx]
        true = true_counts[idx]
        rand_tp = rand_tp_counts[idx]
        rand_pred = rand_pred_counts[idx]
        precision_bg = tp / pred if pred else 0.0
        recall_bg = tp / true if true else 0.0
        f1_bg = 2 * precision_bg * recall_bg / (precision_bg + recall_bg) if (precision_bg + recall_bg) else 0.0
        rand_precision_bg = rand_tp / rand_pred if rand_pred else 0.0
        rand_recall_bg = rand_tp / true if true else 0.0
        rand_f1_bg = (
            2 * rand_precision_bg * rand_recall_bg / (rand_precision_bg + rand_recall_bg)
            if (rand_precision_bg + rand_recall_bg)
            else 0.0
        )
        dataset_pct = dataset_occurrence_counts[idx] / dataset_total_records if dataset_total_records else 0.0
        per_bigram_rows.append(
            {
                "bigram": gram,
                "precision": precision_bg,
                "recall": recall_bg,
                "f1": f1_bg,
                "tp": tp,
                "pred_count": pred,
                "true_count": true,
                "dataset_pct": dataset_pct,
                "rand_precision": rand_precision_bg,
                "rand_recall": rand_recall_bg,
                "rand_f1": rand_f1_bg,
                "rand_tp": rand_tp,
                "rand_pred_count": rand_pred,
            }
        )

    return avg_metrics, results, pd.DataFrame(per_bigram_rows).sort_values(by="dataset_pct", ascending=False)


def save_evaluation_outputs(context, model_bundle, GLOBAL_CONFIG, avg_metrics, results, per_bigram_df, logger):
    per_bigram_df.to_csv(
        os.path.join(context["current_experiment_directory"], "per_bigram_metrics.csv"),
        index=False,
    )

    if GLOBAL_CONFIG["SaveResults"]:
        metrics_df = pd.DataFrame(
            {
                "metric": [
                    "avg_precision",
                    "avg_recall",
                    "avg_f1",
                    "avg_dice",
                    "rand_avg_precision",
                    "rand_avg_recall",
                    "rand_avg_f1",
                    "rand_avg_dice",
                ],
                "value": [
                    avg_metrics["avg_precision"],
                    avg_metrics["avg_recall"],
                    avg_metrics["avg_f1"],
                    avg_metrics["avg_dice"],
                    avg_metrics["rand_avg_precision"],
                    avg_metrics["rand_avg_recall"],
                    avg_metrics["rand_avg_f1"],
                    avg_metrics["rand_avg_dice"],
                ],
            }
        )
        metrics_path = f"{model_bundle['trained_model_directory']}/metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        if logger.isEnabledFor(logging.INFO):
            logger.info("Saved metrics to %s", metrics_path)

    if GLOBAL_CONFIG["SavePredictions"]:
        results_path = f"{model_bundle['trained_model_directory']}/results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        if logger.isEnabledFor(logging.INFO):
            logger.info("Saved predictions to %s", results_path)

    results_df = pd.json_normalize(results)
    plot_metric_distributions(
        results_df,
        model_bundle["trained_model_directory"],
        save=GLOBAL_CONFIG["SaveResults"],
    )
    return results_df


def maybe_run_reidentification(context, GLOBAL_CONFIG, logger, results):
    analysis_data_path = GLOBAL_CONFIG.get("AnalysisData")
    if analysis_data_path:
        if logger.isEnabledFor(logging.INFO):
            logger.info("Skipping re-identification because AnalysisData may not share UIDs with the training splits.")
        return

    header = read_header(GLOBAL_CONFIG["Data"])
    df_not_reid_cached = get_not_reidentified_df(
        context["data_dir"],
        context["identifier"],
        alice_enc_hash=context["alice_enc_hash"],
    )
    run_reidentification_greedy(
        results,
        header,
        df_not_reid_cached,
        current_experiment_directory=context["current_experiment_directory"],
    )
