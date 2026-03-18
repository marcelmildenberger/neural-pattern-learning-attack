import json
import os
from hashlib import md5

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Subset


KEYS_TO_REMOVE = [
    "config", "checkpoint_dir_name", "experiment_tag", "done", "training_iteration",
    "trial_id", "date", "time_this_iter_s", "pid", "time_total_s", "hostname",
    "node_ip", "time_since_restore", "iterations_since_restore", "timestamp",
]


def get_hashes(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG):
    """Build stable cache keys for the encoding and embedding configuration."""
    if GLOBAL_CONFIG["DropFrom"] == "Alice":
        eve_enc_hash = md5(
            ("%s-%s-DropAlice" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()
        ).hexdigest()
        alice_enc_hash = md5(
            ("%s-%s-%s-DropAlice" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"], GLOBAL_CONFIG["Overlap"])).encode()
        ).hexdigest()
        eve_emb_hash = md5(
            ("%s-%s-%s-DropAlice" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()
        ).hexdigest()
        alice_emb_hash = md5(
            (
                "%s-%s-%s-%s-DropAlice"
                % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"], GLOBAL_CONFIG["Overlap"])
            ).encode()
        ).hexdigest()
    elif GLOBAL_CONFIG["DropFrom"] == "Eve":
        eve_enc_hash = md5(
            ("%s-%s-%s-DropEve" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"], GLOBAL_CONFIG["Overlap"])).encode()
        ).hexdigest()
        alice_enc_hash = md5(("%s-%s-DropEve" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()
        eve_emb_hash = md5(
            (
                "%s-%s-%s-%s-DropEve"
                % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"], GLOBAL_CONFIG["Overlap"])
            ).encode()
        ).hexdigest()
        alice_emb_hash = md5(
            ("%s-%s-%s-DropEve" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()
        ).hexdigest()
    else:
        eve_enc_hash = md5(
            ("%s-%s-%s-DropBoth" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"], GLOBAL_CONFIG["Overlap"])).encode()
        ).hexdigest()
        alice_enc_hash = md5(
            ("%s-%s-%s-DropBoth" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"], GLOBAL_CONFIG["Overlap"])).encode()
        ).hexdigest()
        eve_emb_hash = md5(
            (
                "%s-%s-%s-%s-DropBoth"
                % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"], GLOBAL_CONFIG["Overlap"])
            ).encode()
        ).hexdigest()
        alice_emb_hash = md5(
            (
                "%s-%s-%s-%s-DropBoth"
                % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"], GLOBAL_CONFIG["Overlap"])
            ).encode()
        ).hexdigest()

    return eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash


def precision_recall_f1(y_true, y_pred):
    true_set, pred_set = set(y_true), set(y_pred)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1


def run_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    is_training,
    scheduler=None,
    scheduler_step=None,
    clip_grad_norm=0.0,
):
    """Run one train or evaluation epoch and return the average loss."""
    model.train(mode=is_training)
    running_loss = 0.0

    with torch.set_grad_enabled(is_training):
        for data, labels, _ in dataloader:
            data, labels = data.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                if clip_grad_norm and clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                if scheduler is not None and scheduler_step == "batch":
                    scheduler.step()

            running_loss += loss.item() * labels.size(0)

    return running_loss / len(dataloader.dataset)


def map_probabilities_to_bi_grams(bi_gram_dict, probabilities):
    return [
        {bi_gram_dict[j]: prob.item() for j, prob in enumerate(sample)}
        for sample in probabilities
    ]


def filter_high_scoring_bi_grams(bi_gram_scores, threshold, max_grams=33):
    filtered = []
    for score_dict in bi_gram_scores:
        filtered_grams = [(gram, score) for gram, score in score_dict.items() if score > threshold]
        top_grams = sorted(filtered_grams, key=lambda x: x[1], reverse=True)[:max_grams]
        filtered.append([gram for gram, _ in top_grams])
    return filtered


def calculate_performance_metrics(actual_batch, predicted_batch):
    total_precision = total_recall = total_f1 = total_dice = 0.0

    for actual, predicted in zip(actual_batch, predicted_batch):
        precision, recall, f1 = precision_recall_f1(actual, predicted)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_dice += dice_coefficient(actual, predicted)

    return total_dice, total_precision, total_recall, total_f1


def decode_labels_to_bi_grams(bi_gram_dict, label_batch):
    return [
        [bi_gram_dict[i] for i, val in enumerate(label_tensor) if val == 1]
        for label_tensor in label_batch
    ]


def print_and_save_result(label, result, save_to):
    print(f"\n {label}")
    print("-" * 40)

    config = resolve_config(result.config)
    metrics = result.metrics

    print(f"Config: {config}")
    print(f"Average Dice: {metrics.get('average_dice'):.4f}")
    print(f"Average Precision: {metrics.get('average_precision'):.4f}")
    print(f"Average Recall: {metrics.get('average_recall'):.4f}")
    print(f"Average F1: {metrics.get('average_f1'):.4f}")

    result_record = {**config, **metrics}
    clean_result_dict(result_record)

    with open(f"{save_to}/{label}.json", "w") as f:
        json.dump(result_record, f, indent=4)


def clean_result_dict(result_dict):
    for key in KEYS_TO_REMOVE:
        result_dict.pop(key, None)
    return result_dict


def resolve_config(config):
    resolved = {}

    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = resolve_config(value)
        elif isinstance(value, (int, float, str, Subset)):
            resolved[key] = value
        else:
            resolved[key] = value.sample()

    return resolved


def metrics_per_entry(actual, predicted):
    actual_set, predicted_set = set(actual), set(predicted)
    precision, recall, f1 = precision_recall_f1(actual_set, predicted_set)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "actual_len": len(actual_set),
        "predicted_len": len(predicted_set),
        "dice": dice_coefficient(actual_set, predicted_set),
        "jaccard": jaccard_similarity(actual_set, predicted_set),
    }


def dice_coefficient(set1, set2) -> float:
    set1, set2 = set(set1), set(set2)
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    return round(2 * intersection / (len(set1) + len(set2)), 4)


def jaccard_similarity(set1, set2):
    set1, set2 = set(set1), set(set2)
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def to_minutes(seconds):
    return round(seconds / 60, 2)


def save_nepal_runtime_log(
    elapsed_gma,
    elapsed_hyperparameter_optimization,
    elapsed_model_training,
    elapsed_application_to_encoded_data,
    elapsed_refinement_and_reconstruction,
    elapsed_total,
    output_dir="nepal_runtime_logs",
):
    os.makedirs(output_dir, exist_ok=True)

    runtimes = {
        "gma": elapsed_gma,
        "hyperparameter_optimization": elapsed_hyperparameter_optimization,
        "model_training": elapsed_model_training,
        "application_to_encoded_data": elapsed_application_to_encoded_data,
        "refinement_and_reconstruction": elapsed_refinement_and_reconstruction,
        "total_runtime": elapsed_total,
    }

    runtime_data = []
    for label, seconds in runtimes.items():
        runtime_data.append(
            {
                "phase": label,
                "runtime_seconds": seconds,
                "runtime_minutes": to_minutes(seconds),
            }
        )

    pd.DataFrame(runtime_data).to_csv(os.path.join(output_dir, "nepal_runtime_log.csv"), index=False)


def log_epoch_metrics(epoch, total_epochs, train_loss, val_loss, tb_writer=None, save_results=False):
    epoch_str = f"[{epoch + 1}/{total_epochs}]"
    print(f"{epoch_str} Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    if save_results and tb_writer is not None:
        tb_writer.add_scalar("Loss/train", train_loss, epoch + 1)
        tb_writer.add_scalar("Loss/validation", val_loss, epoch + 1)


def plot_loss_curves(train_losses, val_losses, save_path=None, save=False):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training loss", color="blue")
    plt.plot(val_losses, label="Validation loss", color="red")
    plt.legend()
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save and save_path is not None:
        plt.savefig(save_path)
    plt.close()


def plot_metric_distributions(results_df, trained_model_directory, save=False):
    metric_cols = ["precision", "recall", "f1", "dice", "jaccard"]
    melted = results_df.melt(value_vars=metric_cols, var_name="metric", value_name="score")
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=melted,
        x="score",
        hue="metric",
        bins=20,
        element="step",
        fill=False,
        kde=True,
        palette="Set2",
    )
    plt.title("Distribution of Precision / Recall / F1 across Samples")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    if save:
        out_path = os.path.join(trained_model_directory, "metric_distributions.png")
        plt.savefig(out_path)
    plt.close()
