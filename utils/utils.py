# Standard library imports
import csv
import json
import os
from pathlib import Path
from hashlib import md5
from typing import Sequence

# Third-party imports
import hickle as hkl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from tqdm import tqdm
import pickle
from torch.utils.data import random_split, Subset
import seaborn as sns
from utils.encoder_registry import encoded_dataset_path, get_encoder_spec
from utils.string_utils import *
import random


# List of keys to remove
keys_to_remove = [
    "config", "checkpoint_dir_name", "experiment_tag", "done", "training_iteration",
    "trial_id", "date", "time_this_iter_s", "pid", "time_total_s", "hostname",
    "node_ip", "time_since_restore", "iterations_since_restore", "timestamp"
]

def get_cache_path(data_directory, identifier, alice_enc_hash, name="dataset"):
    os.makedirs(f"{data_directory}/cache", exist_ok=True)
    return os.path.join(data_directory, "cache", f"{name}_{identifier}_{alice_enc_hash}.pkl")

def read_tsv(path: str, skip_header: bool = True, as_dict: bool = False, delim: str = "\t") -> Sequence[Sequence[str]]:
    data = {} if as_dict else []
    uid = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delim)
        if skip_header:
            header = next(reader)
        else:
            header = next(reader)
        for row in reader:
            if as_dict:
                assert len(row) == 3, "Dict mode only supports rows with two values + uid"
                data[row[0]] = row[1]
            else:
                data.append(row[:-1])
                uid.append(row[-1])
    return data, uid, header


def save_tsv(data, path: str, delim: str = "\t", mode="w", write_header: bool = False, header: list[str]= None):
    with open(path, mode, newline="") as f:
        csvwriter = csv.writer(f, delimiter=delim)
        if(write_header):
            csvwriter.writerow(header)
        csvwriter.writerows(data)


def get_hashes(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG):
     # Compute hashes of configuration to store/load data and thus avoid redundant computations.
    # Using MD5 because Python's native hash() is not stable across processes
    if GLOBAL_CONFIG["DropFrom"] == "Alice":

        eve_enc_hash = md5(
            ("%s-%s-DropAlice" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()
        alice_enc_hash = md5(
            ("%s-%s-%s-DropAlice" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                     GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        eve_emb_hash = md5(
            ("%s-%s-%s-DropAlice" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()

        alice_emb_hash = md5(("%s-%s-%s-%s-DropAlice" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                         GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()
    elif GLOBAL_CONFIG["DropFrom"] == "Eve":

        eve_enc_hash = md5(
            ("%s-%s-%s-DropEve" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                   GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_enc_hash = md5(("%s-%s-DropEve" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()

        eve_emb_hash = md5(("%s-%s-%s-%s-DropEve" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                     GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_emb_hash = md5(("%s-%s-%s-DropEve" % (str(EMB_CONFIG), str(ENC_CONFIG),
                                                    GLOBAL_CONFIG["Data"])).encode()).hexdigest()
    else:
        eve_enc_hash = md5(
            ("%s-%s-%s-DropBoth" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                    GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_enc_hash = md5(
            ("%s-%s-%s-DropBoth" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                    GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        eve_emb_hash = md5(("%s-%s-%s-%s-DropBoth" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                      GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_emb_hash = md5(("%s-%s-%s-%s-DropBoth" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                        GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

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


def run_epoch(model, dataloader, criterion, optimizer, device, is_training, verbose, scheduler=None, scheduler_step=None, clip_grad_norm=0.0):
    model.train(mode=is_training)
    running_loss = 0.0

    data_iter = tqdm(dataloader, desc="Training" if is_training else "Validation") if verbose else dataloader

    with torch.set_grad_enabled(is_training):
        for data, labels, _ in data_iter:
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
        # Filter by threshold
        filtered_grams = [(gram, score) for gram, score in score_dict.items() if score > threshold]
        # Sort by score descending and keep only top `max_grams`
        top_grams = sorted(filtered_grams, key=lambda x: x[1], reverse=True)[:max_grams]
        # Extract just the 2-grams
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

    # Save as JSON for better analysis
    with open(f"{save_to}/{label}.json", 'w') as f:
        json.dump(result_record, f, indent=4)


def clean_result_dict(result_dict):
    for key in keys_to_remove:
        result_dict.pop(key, None)
    return result_dict

def resolve_config(config):
    resolved = {}

    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively resolve nested dictionaries
            resolved[key] = resolve_config(value)
        elif isinstance(value, (int, float, str, Subset)):
            # Use the value as-is if it's a basic type or Subset
            resolved[key] = value
        else:
            # Assume it's a Ray Tune search space object and sample a concrete value
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


def dice_coefficient(set1: set, set2: set) -> float:
    set1, set2 = set(set1), set(set2)
    if not set1 and not set2:
        return 1.0  # both empty sets → full similarity
    intersection = len(set1 & set2)
    return round(2 * intersection / (len(set1) + len(set2)), 4)


def jaccard_similarity(set1, set2):
    set1, set2 = set(set1), set(set2)
    if not set1 and not set2:
        return 1.0  # both empty sets → full similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def greedy_reconstruction(results):
    reconstructed_results = []

    for entry in results:
        uid = entry["uid"]
        bi_grams = entry["predicted_bi_grams"]


        # Build directed graph from 2-grams
        G = nx.DiGraph()
        G.add_edges_from((gram[0], gram[1]) for gram in bi_grams)

        if nx.is_directed_acyclic_graph(G):
            path = nx.dag_longest_path(G)
            reconstructed = path[0] + ''.join(path[1:]) if path else ""
        else:
            # DFS to find longest possible sequence
            def dfs(node, visited_edges, current_string):
                nonlocal longest_sequence
                if len(current_string) > len(longest_sequence):
                    longest_sequence = current_string

                for neighbor in G.successors(node):
                    edge = (node, neighbor)
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        dfs(neighbor, visited_edges, current_string + neighbor)
                        visited_edges.remove(edge)

            longest_sequence = ""
            for gram in bi_grams:
                dfs(gram[1], {(gram[0], gram[1])}, gram[0] + gram[1])

            reconstructed = longest_sequence

        reconstructed_results.append({
            "uid": uid,
            "identifier": reconstructed
        })

    return reconstructed_results


def create_identifier_column_dynamic(df, components):
    cleaned_cols = [
        df[col].astype(str).str.replace('/', '', regex=False)
        for col in components
    ]
    return pd.Series(map(''.join, zip(*cleaned_cols)), index=df.index).str.lower()


def reidentification_analysis(df_1, df_2, merge_on, len_not_reidentified, save_path=None):
    merged = pd.merge(df_1, df_2, on=merge_on, how='inner', suffixes=('_df1', '_df2'))

    total_reidentified = len(merged)
    total_not_reidentified = len_not_reidentified

    print("Reidentification Analysis:")
    print(f"Total Reidentified Individuals: {total_reidentified}")
    print(f"Total Not Reidentified Individuals: {total_not_reidentified}")

    if total_not_reidentified > 0:
        reidentification_rate = (total_reidentified / total_not_reidentified) * 100
        print(f"Reidentification Rate: {reidentification_rate:.2f}%")
    else:
        reidentification_rate = None
        print("No not reidentified individuals to analyze.")

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Save merged reidentified individuals
        result_csv_path = os.path.join(save_path, f"result_greedy.csv")
        merged.to_csv(result_csv_path, index=False)

        # Save summary as CSV for better analysis
        summary_data = {
            "metric": ["reidentification_method", "total_reidentified_individuals", "total_not_reidentified_individuals", "reidentification_rate"],
            "value": ['greedy', total_reidentified, total_not_reidentified, f"{reidentification_rate:.2f}%" if reidentification_rate is not None else "N/A"]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(save_path, f"summary_greedy.csv")
        summary_df.to_csv(summary_csv_path, index=False)

    return merged


# Convert seconds to minutes
def to_minutes(seconds):
    return round(seconds / 60, 2)


def save_nepal_runtime_log(
    elapsed_gma,
    elapsed_hyperparameter_optimization,
    elapsed_model_training,
    elapsed_application_to_encoded_data,
    elapsed_refinement_and_reconstruction,
    elapsed_total,
    output_dir="nepal_runtime_logs"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV for better analysis
    runtimes = {
        "gma": elapsed_gma,
        "hyperparameter_optimization": elapsed_hyperparameter_optimization,
        "model_training": elapsed_model_training,
        "application_to_encoded_data": elapsed_application_to_encoded_data,
        "refinement_and_reconstruction": elapsed_refinement_and_reconstruction,
        "total_runtime": elapsed_total
    }
    
    # Convert to DataFrame and save as CSV
    runtime_data = []
    for label, seconds in runtimes.items():
        runtime_data.append({
            "phase": label,
            "runtime_seconds": seconds,
            "runtime_minutes": to_minutes(seconds)
        })
    
    runtime_df = pd.DataFrame(runtime_data)
    runtime_df.to_csv(os.path.join(output_dir, "nepal_runtime_log.csv"), index=False)


def load_dataframe(path):
    data = hkl.load(path)
    return pd.DataFrame(data[1:], columns=data[0])


def read_header(tsv_path):
    path = Path(tsv_path)

    # If the noisy copy is missing, try the parent datasets directory.
    if not path.exists() and path.parent.name == "noisy":
        fallback = path.parent.parent / path.name
        if fallback.exists():
            path = fallback

    if not path.exists():
        raise FileNotFoundError(f"TSV file not found for header: {tsv_path}")

    with path.open('r', encoding='utf-8') as f:
        header_line = f.readline().strip()
        columns = header_line.split('\t')
        return columns


def create_synthetic_data_splits(GLOBAL_CONFIG, ENC_CONFIG, data_dir, alice_enc_hash, identifier, 
                                path_reidentified, path_not_reidentified, path_all):
    """
    Create synthetic data splits when GraphMatchingAttack is disabled.
    Loads the encoded dataset and samples based on overlap percentage.
    """
    
    import os

    # Load the encoded dataset
    data_path = GLOBAL_CONFIG["Data"]          # e.g. "./data/datasets/noisy/fakename_1k.tsv"

    encoded_file = encoded_dataset_path(
        data_path,
        ENC_CONFIG["AliceAlgo"],
        diffuse=ENC_CONFIG.get("AliceDiffuse", False),
    )

    if not os.path.isfile(encoded_file):
        raise FileNotFoundError(f"Encoded dataset not found: {encoded_file}")

    print("Loading Dataset: " + encoded_file)
    # Load the encoded data
    data, uids, header = read_tsv(encoded_file, skip_header=True, as_dict=False)

    # Reconstruct full data rows by re-attaching the uid as the last column.
    # This works for any dataset shape (with or without a birthday column).
    all_data = [header] + [row + [uid] for row, uid in zip(data, uids)]
    
    # Sample based on overlap percentage
    overlap_ratio = GLOBAL_CONFIG["Overlap"]
    n_total = len(all_data) - 1  # Subtract header
    n_reidentified = int(n_total * overlap_ratio)
    
    # Random sampling
    indices = list(range(1, len(all_data)))  # Skip header
    reidentified_indices = random.sample(indices, n_reidentified)
    not_reidentified_indices = [i for i in indices if i not in reidentified_indices]
    
    # Create reidentified data (training data) - full format for training
    reidentified_data = [all_data[0]]  # Full header
    for idx in reidentified_indices:
        reidentified_data.append(all_data[idx])
    
    not_reidentified_header = [header[-2], header[-1]]  # Last 2 column names: encoding, uid
    not_reidentified_data = [not_reidentified_header]
    for idx in not_reidentified_indices:
        # Extract only the last 2 columns (encoding + uid) to match GMA's alice_entry format
        row = all_data[idx]
        encoding_and_uid = [row[-2], row[-1]]  # Last 2 columns: encoding, uid
        not_reidentified_data.append(encoding_and_uid)
    
    # Save the synthetic splits
    os.makedirs(os.path.dirname(path_reidentified), exist_ok=True)
    os.makedirs(os.path.dirname(path_not_reidentified), exist_ok=True)
    os.makedirs(os.path.dirname(path_all), exist_ok=True)
    
    hkl.dump(reidentified_data, path_reidentified, mode="w")
    hkl.dump(not_reidentified_data, path_not_reidentified, mode="w")
    hkl.dump(all_data, path_all, mode="w")


def load_experiment_datasets(
    data_directory, alice_enc_hash, identifier, ENC_CONFIG, nepal_CONFIG, GLOBAL_CONFIG, all_bi_grams, splits=("train", "val", "test")
):
    cache_path = get_cache_path(data_directory, identifier, alice_enc_hash)
    # Try to load from cache if all splits are present
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
            return {k: cached.get(k) for k in splits}

    # Otherwise, create datasets
    df_reidentified = load_dataframe(f"{data_directory}/available_to_eve/reidentified_individuals_{identifier}.h5")

    df_not_reidentified = load_dataframe(f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5")
    df_all = load_dataframe(f"{data_directory}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5")
    df_test = df_all[df_all["uid"].isin(df_not_reidentified["uid"])].reset_index(drop=True)

    encoder_spec = get_encoder_spec(ENC_CONFIG["AliceAlgo"])
    DatasetClass = encoder_spec.load_dataset_class()

    if encoder_spec.needs_integer_vocabulary:
        # Calculate unique integers from the complete dataset to ensure consistent tensor dimensions
        # Using df_all ensures we capture all possible hash values that could appear in any subset
        
        # Parse the string representation of sets to extract actual integers
        def parse_twostephash_string(twostephash_str):
            # Remove curly braces and split by comma
            # Handle both string format "{1, 2, 3}" and actual set objects
            if isinstance(twostephash_str, str):
                # Remove curly braces and split by comma
                content = twostephash_str.strip('{}')
                if content:  # Handle empty sets
                    return [int(x.strip()) for x in content.split(',')]
                else:
                    return []
            else:
                # If it's already a set or list, convert to list of ints
                return [int(x) for x in twostephash_str]
        
        # Extract all unique integers from all twostephash entries
        all_ints = []
        for twostephash_entry in df_all[encoder_spec.column_name]:
            all_ints.extend(parse_twostephash_string(twostephash_entry))
        
        unique_ints = sorted(set(all_ints))
        dataset_args = {"all_integers": unique_ints}
    else:
        dataset_args = {}
    common_args = {
        "is_labeled": True,
        "all_bi_grams": all_bi_grams,
        "dev_mode": GLOBAL_CONFIG["DevMode"]
    }

    data_labeled = DatasetClass(df_reidentified, **common_args, **dataset_args)
    data_test = DatasetClass(df_test, **common_args, **dataset_args)
    train_size = int(nepal_CONFIG["TrainSize"] * len(data_labeled))
    val_size = len(data_labeled) - train_size
    data_train, data_val = random_split(data_labeled, [train_size, val_size])
    result = {"train": data_train, "val": data_val, "test": data_test}
    # Save all splits to cache for future use
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    # Return only the requested splits/datasets
    return {k: result[k] for k in splits}

def load_not_reidentified_data(data_directory, alice_enc_hash, identifier):
    cache_path = get_cache_path(data_directory, identifier, alice_enc_hash, name="not_reidentified")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            df_filtered = pickle.load(f)
        return df_filtered
    df_not_reidentified = load_dataframe(f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5")
    df_all = load_dataframe(f"{data_directory}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5")
    df_filtered = df_all[df_all["uid"].isin(df_not_reidentified["uid"])].reset_index(drop=True)
    drop_col = df_filtered.columns[-2]
    df_filtered = df_filtered.drop(columns=[drop_col])
    with open(cache_path, 'wb') as f:
        pickle.dump(df_filtered, f)
    return df_filtered

def get_not_reidentified_df(data_dir: str, identifier: str, alice_enc_hash=None) -> pd.DataFrame:
    """
    Loads and lowercases the DataFrame of not re-identified individuals.
    Args:
        data_dir (str): Path to the data directory.
        identifier (str): Unique identifier for the experiment/data split.
        alice_enc_hash (str, optional): Hash for Alice's encoding. If not provided, must be passed by caller.
    Returns:
        pd.DataFrame: Lowercased DataFrame of not re-identified individuals.
    """
    if alice_enc_hash is None:
        raise ValueError("alice_enc_hash must be provided.")
    df = load_not_reidentified_data(data_dir, alice_enc_hash, identifier)
    return lowercase_df(df)


def create_identifier(df: pd.DataFrame, components):
    """
    Adds an 'identifier' column to the DataFrame using the specified components.
    Args:
        df (pd.DataFrame): Input DataFrame.
        components (list): List of column names to use for identifier creation.
    Returns:
        pd.DataFrame: DataFrame with 'uid' and 'identifier' columns.
    """
    df = df.copy()
    df["identifier"] = create_identifier_column_dynamic(df, components)
    return df[["uid", "identifier"]]


def run_reidentification_greedy(results, header, df_not_reidentified, current_experiment_directory):
    
    
    reconstructed_identities = greedy_reconstruction(results)

    df_reconstructed = lowercase_df(pd.DataFrame(reconstructed_identities, columns=["uid", "identifier"]))
    # If identifier components are provided, create identifier column in not-reidentified DataFrame
    df_not_reidentified = create_identifier(df_not_reidentified, header[:-1])
    return reidentification_analysis(
        df_reconstructed,
        df_not_reidentified,
        ["uid", "identifier"],
        len(df_not_reidentified),
        save_path=f"{current_experiment_directory}/re_identification_results"
    )
    

def log_epoch_metrics(epoch, total_epochs, train_loss, val_loss, tb_writer=None, save_results=False):
    """
    Log and optionally write train/val loss to TensorBoard for a given epoch.
    Args:
        epoch: Current epoch (int)
        total_epochs: Total number of epochs (int)
        train_loss: Training loss (float)
        val_loss: Validation loss (float)
        tb_writer: TensorBoard SummaryWriter (optional)
        save_results: Whether to save results to TensorBoard (bool)
    """
    epoch_str = f"[{epoch + 1}/{total_epochs}]"
    print(f"{epoch_str} Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    if save_results and tb_writer is not None:
        tb_writer.add_scalar("Loss/train", train_loss, epoch + 1)
        tb_writer.add_scalar("Loss/validation", val_loss, epoch + 1)


def plot_loss_curves(train_losses, val_losses, save_path=None, save=False):
    """
    Plot training and validation loss curves. Optionally save to file if save=True.
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot (if save=True)
        save: Whether to save the plot to file
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss', color='blue')
    plt.plot(val_losses, label='Validation loss', color='red')
    plt.legend()
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save and save_path is not None:
        plt.savefig(save_path)
    plt.close()

def plot_metric_distributions(results_df, trained_model_directory, save=False):
    """
    Plot and optionally save the distribution of precision, recall, F1, dice, and jaccard metrics.
    Args:
        results_df: DataFrame with per-sample metrics
        trained_model_directory: Directory to save the plot if save=True
        save: Whether to save the plot
    """
    metric_cols = ["precision", "recall", "f1", "dice", "jaccard"]
    melted = results_df.melt(value_vars=metric_cols,
                            var_name="metric",
                            value_name="score")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=melted,
                x="score",
                hue="metric",
                bins=20,
                element="step",
                fill=False,
                kde=True,
                palette="Set2")
    plt.title("Distribution of Precision / Recall / F1 across Samples")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    if save:
        out_path = os.path.join(trained_model_directory, "metric_distributions.png")
        plt.savefig(out_path)
    plt.close()
