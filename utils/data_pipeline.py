import csv
import os
import pickle
import random
from pathlib import Path
from typing import Sequence

import hickle as hkl
import networkx as nx
import pandas as pd
from pytorch_datasets.bloom_filter_dataset import BloomFilterDataset
from pytorch_datasets.roundbased_encoding_scheme_dataset import RoundBasedEncodingSchemeDataset
from pytorch_datasets.tab_min_hash_dataset import TabMinHashDataset
from pytorch_datasets.two_step_hash_dataset import TwoStepHashDataset
from torch.utils.data import random_split

from utils.string_utils import lowercase_df, normalize_alphanumeric


def get_cache_path(data_directory, identifier, alice_enc_hash, name="dataset", extra_key=""):
    """Return a cache path that is sensitive to the current experiment setup."""
    os.makedirs(f"{data_directory}/cache", exist_ok=True)
    suffix = f"_{extra_key}" if extra_key else ""
    return os.path.join(data_directory, "cache", f"{name}_{identifier}_{alice_enc_hash}{suffix}.pkl")


def read_tsv(path: str, skip_header: bool = True, as_dict: bool = False, delim: str = "\t") -> Sequence[Sequence[str]]:
    data = {} if as_dict else []
    uid = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)
        for row in reader:
            if as_dict:
                assert len(row) == 3, "Dict mode only supports rows with two values + uid"
                data[row[0]] = row[1]
            else:
                data.append(row[:-1])
                uid.append(row[-1])
    return data, uid, header


def save_tsv(data, path: str, delim: str = "\t", mode="w", write_header: bool = False, header=None):
    with open(path, mode, newline="") as f:
        csvwriter = csv.writer(f, delimiter=delim)
        if write_header:
            csvwriter.writerow(header)
        csvwriter.writerows(data)


def greedy_reconstruction(results):
    reconstructed_results = []

    for entry in results:
        uid = entry["uid"]
        bi_grams = entry["predicted_bi_grams"]

        graph = nx.DiGraph()
        graph.add_edges_from((gram[0], gram[1]) for gram in bi_grams)

        if nx.is_directed_acyclic_graph(graph):
            path = nx.dag_longest_path(graph)
            reconstructed = path[0] + "".join(path[1:]) if path else ""
        else:
            def dfs(node, visited_edges, current_string):
                nonlocal longest_sequence
                if len(current_string) > len(longest_sequence):
                    longest_sequence = current_string

                for neighbor in graph.successors(node):
                    edge = (node, neighbor)
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        dfs(neighbor, visited_edges, current_string + neighbor)
                        visited_edges.remove(edge)

            longest_sequence = ""
            for gram in bi_grams:
                dfs(gram[1], {(gram[0], gram[1])}, gram[0] + gram[1])

            reconstructed = longest_sequence

        reconstructed_results.append({"uid": uid, "identifier": reconstructed})

    return reconstructed_results


def create_identifier_column_dynamic(df, components):
    cleaned_cols = [df[col].astype(str).map(normalize_alphanumeric) for col in components]
    return pd.Series(map("".join, zip(*cleaned_cols)), index=df.index).str.lower()


def reidentification_analysis(df_1, df_2, merge_on, len_not_reidentified, save_path=None):
    merged = pd.merge(df_1, df_2, on=merge_on, how="inner", suffixes=("_df1", "_df2"))

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
        merged.to_csv(os.path.join(save_path, "result_greedy.csv"), index=False)

        summary_data = {
            "metric": [
                "reidentification_method",
                "total_reidentified_individuals",
                "total_not_reidentified_individuals",
                "reidentification_rate",
            ],
            "value": [
                "greedy",
                total_reidentified,
                total_not_reidentified,
                f"{reidentification_rate:.2f}%" if reidentification_rate is not None else "N/A",
            ],
        }
        pd.DataFrame(summary_data).to_csv(os.path.join(save_path, "summary_greedy.csv"), index=False)

    return merged


def load_dataframe(path):
    data = hkl.load(path)
    return pd.DataFrame(data[1:], columns=data[0])


def read_header(tsv_path):
    path = Path(tsv_path)

    if not path.exists() and path.parent.name == "noisy":
        fallback = path.parent.parent / path.name
        if fallback.exists():
            path = fallback

    if not path.exists():
        raise FileNotFoundError(f"TSV file not found for header: {tsv_path}")

    with path.open("r", encoding="utf-8") as f:
        return f.readline().strip().split("\t")


def resolve_encoded_dataset_path(data_path: str, algo: str, diffuse: bool = False) -> str:
    base_path, _ = os.path.splitext(data_path)

    if algo == "BloomFilter":
        suffix = "_bfd_encoded.tsv" if diffuse else "_bf_encoded.tsv"
    elif algo == "TabMinHash":
        suffix = "_tmh_encoded.tsv"
    elif algo == "TwoStepHash":
        suffix = "_tsh_encoded.tsv"
    elif algo in {"Saul", "RoundBasedEncoder"}:
        suffix = "_saul_encoded.tsv"
    elif algo == "RSE":
        suffix = "_rse_encoded.tsv"
    else:
        raise ValueError(f"Unsupported encoding algorithm: {algo}")

    return f"{base_path}{suffix}"


def create_synthetic_data_splits(
    GLOBAL_CONFIG,
    ENC_CONFIG,
    data_dir,
    alice_enc_hash,
    identifier,
    path_reidentified,
    path_not_reidentified,
    path_all,
):
    encoded_file = resolve_encoded_dataset_path(
        GLOBAL_CONFIG["Data"],
        ENC_CONFIG["AliceAlgo"],
        diffuse=ENC_CONFIG.get("AliceDiffuse", False),
    )

    if not os.path.isfile(encoded_file):
        raise FileNotFoundError(f"Encoded dataset not found: {encoded_file}")

    print("Loading Dataset: " + encoded_file)
    data, uids, header = read_tsv(encoded_file, skip_header=True, as_dict=False)
    all_data = [header] + [row + [uid] for row, uid in zip(data, uids)]

    overlap_ratio = GLOBAL_CONFIG["Overlap"]
    n_total = len(all_data) - 1
    n_reidentified = int(n_total * overlap_ratio)

    indices = list(range(1, len(all_data)))
    rnd = random.Random(int(alice_enc_hash[:8], 16))
    reidentified_indices = rnd.sample(indices, n_reidentified)
    not_reidentified_indices = [i for i in indices if i not in reidentified_indices]

    reidentified_data = [all_data[0]]
    for idx in reidentified_indices:
        reidentified_data.append(all_data[idx])

    not_reidentified_header = [header[-2], header[-1]]
    not_reidentified_data = [not_reidentified_header]
    for idx in not_reidentified_indices:
        row = all_data[idx]
        not_reidentified_data.append([row[-2], row[-1]])

    os.makedirs(os.path.dirname(path_reidentified), exist_ok=True)
    os.makedirs(os.path.dirname(path_not_reidentified), exist_ok=True)
    os.makedirs(os.path.dirname(path_all), exist_ok=True)

    hkl.dump(reidentified_data, path_reidentified, mode="w")
    hkl.dump(not_reidentified_data, path_not_reidentified, mode="w")
    hkl.dump(all_data, path_all, mode="w")


def export_gma_results(data_dir, identifier, alice_enc_hash, eve_enc_hash, GLOBAL_CONFIG, output_dir):
    gma_output_dir = Path(output_dir) / "gma_results"
    gma_output_dir.mkdir(parents=True, exist_ok=True)

    reidentified_path = Path(data_dir) / "available_to_eve" / f"reidentified_individuals_{identifier}.h5"
    not_reidentified_path = Path(data_dir) / "available_to_eve" / f"not_reidentified_individuals_{identifier}.h5"

    if not reidentified_path.exists() or not not_reidentified_path.exists():
        raise FileNotFoundError(
            f"GMA artifacts missing for identifier {identifier}. Expected {reidentified_path} and {not_reidentified_path}."
        )

    df_reidentified = load_dataframe(reidentified_path)
    df_not_reidentified = load_dataframe(not_reidentified_path)

    df_reidentified.to_csv(gma_output_dir / "reidentified_individuals.tsv", sep="\t", index=False)
    df_not_reidentified.to_csv(gma_output_dir / "not_reidentified_individuals.tsv", sep="\t", index=False)

    repo_root = Path(__file__).resolve().parents[1]
    alice_encoded_path = repo_root / "graphMatching" / "data" / "encoded" / f"alice-{alice_enc_hash}.h5"
    eve_encoded_path = repo_root / "graphMatching" / "data" / "encoded" / f"eve-{eve_enc_hash}.h5"

    alice_meta = hkl.load(str(alice_encoded_path))
    eve_meta = hkl.load(str(eve_encoded_path))
    alice_record_count = int(alice_meta[0][2])
    eve_record_count = int(eve_meta[0][2])

    overlap_count = None
    denominator = min(alice_record_count, eve_record_count)
    if GLOBAL_CONFIG["DropFrom"] == "Both":
        overlap_path = repo_root / "graphMatching" / "data" / "encoded" / f"overlap-{alice_enc_hash}.pck"
        with open(overlap_path, "rb") as f:
            overlap_count = int(pickle.load(f))
        denominator = overlap_count

    correct_matches = len(df_reidentified)
    success_rate = correct_matches / denominator if denominator else None

    summary_rows = [
        {"metric": "identifier", "value": identifier},
        {"metric": "drop_from", "value": GLOBAL_CONFIG["DropFrom"]},
        {"metric": "overlap_parameter", "value": GLOBAL_CONFIG["Overlap"]},
        {"metric": "matching", "value": GLOBAL_CONFIG["Matching"]},
        {"metric": "matching_metric", "value": GLOBAL_CONFIG["MatchingMetric"]},
        {"metric": "alice_record_count", "value": alice_record_count},
        {"metric": "eve_record_count", "value": eve_record_count},
        {"metric": "overlap_count", "value": overlap_count if overlap_count is not None else ""},
        {"metric": "correct_matches", "value": correct_matches},
        {"metric": "total_reidentified_individuals", "value": len(df_reidentified)},
        {"metric": "total_not_reidentified_individuals", "value": len(df_not_reidentified)},
        {"metric": "success_rate", "value": success_rate if success_rate is not None else ""},
    ]
    pd.DataFrame(summary_rows).to_csv(gma_output_dir / "summary.csv", index=False)


def _dataset_class_for_algo(algo):
    if algo == "BloomFilter":
        return BloomFilterDataset
    if algo == "TabMinHash":
        return TabMinHashDataset
    if algo == "TwoStepHash":
        return TwoStepHashDataset
    if algo in {"RoundBasedEncoder", "Saul", "RSE"}:
        return RoundBasedEncodingSchemeDataset
    raise ValueError(f"Unsupported dataset class for encoding algorithm: {algo}")


def _dataset_args_for_algo(df_all, algo):
    if algo != "TwoStepHash":
        return {}

    def parse_twostephash_string(twostephash_value):
        if isinstance(twostephash_value, str):
            content = twostephash_value.strip("{}")
            return [int(x.strip()) for x in content.split(",")] if content else []
        return [int(x) for x in twostephash_value]

    all_ints = []
    for twostephash_entry in df_all["twostephash"]:
        all_ints.extend(parse_twostephash_string(twostephash_entry))

    return {"all_integers": sorted(set(all_ints))}


def load_experiment_datasets(
    data_directory,
    alice_enc_hash,
    identifier,
    ENC_CONFIG,
    nepal_CONFIG,
    GLOBAL_CONFIG,
    all_bi_grams,
    splits=("train", "val", "test"),
):
    cache_disambiguator = f"train{nepal_CONFIG['TrainSize']}_dev{GLOBAL_CONFIG['DevMode']}"
    cache_path = get_cache_path(data_directory, identifier, alice_enc_hash, extra_key=cache_disambiguator)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return {k: cached.get(k) for k in splits}

    df_reidentified = load_dataframe(f"{data_directory}/available_to_eve/reidentified_individuals_{identifier}.h5")
    df_not_reidentified = load_dataframe(f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5")
    df_all = load_dataframe(f"{data_directory}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5")
    df_test = df_all[df_all["uid"].isin(df_not_reidentified["uid"])].reset_index(drop=True)

    algo = ENC_CONFIG["AliceAlgo"]
    dataset_class = _dataset_class_for_algo(algo)
    dataset_args = _dataset_args_for_algo(df_all, algo)
    common_args = {
        "is_labeled": True,
        "all_bi_grams": all_bi_grams,
        "dev_mode": GLOBAL_CONFIG["DevMode"],
    }

    data_labeled = dataset_class(df_reidentified, **common_args, **dataset_args)
    data_test = dataset_class(df_test, **common_args, **dataset_args)
    train_size = int(nepal_CONFIG["TrainSize"] * len(data_labeled))
    val_size = len(data_labeled) - train_size
    data_train, data_val = random_split(data_labeled, [train_size, val_size])
    result = {"train": data_train, "val": data_val, "test": data_test}

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return {k: result[k] for k in splits}


def load_not_reidentified_data(data_directory, alice_enc_hash, identifier):
    cache_path = get_cache_path(data_directory, identifier, alice_enc_hash, name="not_reidentified")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df_not_reidentified = load_dataframe(f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5")
    df_all = load_dataframe(f"{data_directory}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5")
    df_filtered = df_all[df_all["uid"].isin(df_not_reidentified["uid"])].reset_index(drop=True)
    drop_col = df_filtered.columns[-2]
    df_filtered = df_filtered.drop(columns=[drop_col])

    with open(cache_path, "wb") as f:
        pickle.dump(df_filtered, f)

    return df_filtered


def get_not_reidentified_df(data_dir: str, identifier: str, alice_enc_hash=None) -> pd.DataFrame:
    if alice_enc_hash is None:
        raise ValueError("alice_enc_hash must be provided.")
    df = load_not_reidentified_data(data_dir, alice_enc_hash, identifier)
    return lowercase_df(df)


def create_identifier(df: pd.DataFrame, components):
    df = df.copy()
    df["identifier"] = create_identifier_column_dynamic(df, components)
    return df[["uid", "identifier"]]


def run_reidentification_greedy(results, header, df_not_reidentified, current_experiment_directory):
    reconstructed_identities = greedy_reconstruction(results)
    df_reconstructed = lowercase_df(pd.DataFrame(reconstructed_identities, columns=["uid", "identifier"]))
    df_not_reidentified = create_identifier(df_not_reidentified, header[:-1])
    return reidentification_analysis(
        df_reconstructed,
        df_not_reidentified,
        ["uid", "identifier"],
        len(df_not_reidentified),
        save_path=f"{current_experiment_directory}/re_identification_results",
    )
