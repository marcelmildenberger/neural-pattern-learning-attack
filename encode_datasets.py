"""
Batch-encode all non-encoded TSV datasets under data/datasets (and subfolders) using
the same encoder settings used during the GMA runs. Produces *_bf_encoded.tsv,
*_tmh_encoded.tsv, *_tsh_encoded.tsv and *_rse_encoded.tsv files with the
encoder column inserted right before the uid column.

Example:
python encode_datasets.py --source-dir data/datasets --recursive
"""

import argparse
import copy
import csv
import importlib.util
import itertools
import random
import string
import sys
from collections import Counter
from functools import lru_cache
from hashlib import md5
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


# Defaults mirror experiment_setup.py / nepal_config.json (Alice* values).
DEFAULT_SECRET = "SuperSecretSalt1337"
DEFAULT_NGRAM_SIZE = 2
DEFAULT_BF_LENGTH = 1024
DEFAULT_BF_BITS = 10
DEFAULT_BF_T = 10
DEFAULT_BF_ELD_LENGTH = 1024
DEFAULT_TMH_NUM_HASH = 1024
DEFAULT_TMH_HASH_BITS = 64
DEFAULT_TMH_SUBKEYS = 8
DEFAULT_TMH_ONE_BIT = True
DEFAULT_TSH_NUM_HASH_FUNC = 10
DEFAULT_TSH_NUM_HASH_COL = 1000
DEFAULT_TSH_RAND_MODE = "PNG"
DEFAULT_RSE_K = 40
DEFAULT_RSE_REF_SET_LENGTH = 20
RSE_ALPHABET = string.ascii_lowercase + string.digits


def iter_plain_datasets(root: Path, recursive: bool) -> Iterable[Path]:
    if root.is_file():
        if root.suffix == ".tsv" and not any(
            tag in root.name for tag in ("_bf_encoded", "_tmh_encoded", "_tsh_encoded", "_bfd_encoded", "_rse_encoded", "_saul_encoded")
        ):
            yield root
        return

    pattern = "**/*.tsv" if recursive else "*.tsv"
    for path in root.glob(pattern):
        name = path.name
        if any(tag in name for tag in ("_bf_encoded", "_tmh_encoded", "_tsh_encoded", "_bfd_encoded", "_rse_encoded", "_saul_encoded")):
            continue
        if name.endswith("_analysis.txt"):
            continue
        yield path


def write_tsv(header: Sequence[str], rows: Sequence[Sequence], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for row in rows:
            writer.writerow([str(val) for val in row])


def read_tsv(path: str, skip_header: bool = True, delim: str = "\t") -> tuple[List[List[str]], List[str], List[str]]:
    data: List[List[str]] = []
    uid: List[str] = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)
        for row in reader:
            data.append(row[:-1])
            uid.append(row[-1])
    return data, uid, header


@lru_cache(maxsize=1)
def load_rse_encoder_module():
    encoder_dir = Path(__file__).resolve().parent / "rse-for-pprl" / "encoder"
    module_path = encoder_dir / "data_encoder.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"RSE encoder module not found: {module_path}")

    sys.path.insert(0, str(encoder_dir))
    try:
        spec = importlib.util.spec_from_file_location("rse_data_encoder", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load RSE encoder module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except ModuleNotFoundError as exc:
        if exc.name == "bitarray":
            raise ModuleNotFoundError(
                "RSE encoding requires the 'bitarray' package. Install the project requirements before using '--encoders rse'."
            ) from exc
        raise
    finally:
        if sys.path and sys.path[0] == str(encoder_dir):
            sys.path.pop(0)


@lru_cache(maxsize=1)
def load_rse_ref_set_generator_module():
    generator_dir = Path(__file__).resolve().parent / "rse-for-pprl" / "ref-set-generator"
    module_path = generator_dir / "generator.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"RSE reference-set generator module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("rse_ref_set_generator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RSE reference-set generator from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_vah_hardening_module():
    vah_dir = Path(__file__).resolve().parent / "vah-for-pprl"
    module_path = vah_dir / "hardening.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"VAH hardening module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("vah_hardening", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load VAH hardening module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_rse_value(value: str) -> str:
    normalized = str(value).strip().lower()
    return "".join(ch for ch in normalized if ch.isalnum())


def iter_rse_q_grams(value: str, q: int) -> List[str]:
    normalized = normalize_rse_value(value)
    if len(normalized) < q:
        return []
    return [normalized[i:i + q] for i in range(len(normalized) - (q - 1))]


def extract_rse_q_grams(value: str, q: int) -> set[str]:
    return set(iter_rse_q_grams(value, q))


def build_rse_record_store(data: List[List[str]], uids: List[str], q: int):
    rse = load_rse_encoder_module()
    record_store = {}
    filtered_rows: List[List[str]] = []
    filtered_uids: List[str] = []
    q_gram_counter: Counter[str] = Counter()
    skipped_rows = 0

    for row, uid in zip(data, uids):
        record_q_grams = set()
        missing_value = False

        for value in row:
            if str(value).strip() == "":
                missing_value = True
                break
            record_q_grams.update(extract_rse_q_grams(value, q))

        if missing_value or not record_q_grams:
            skipped_rows += 1
            continue

        normalized_uid = str(uid).strip()
        if normalized_uid in record_store:
            raise ValueError(f"Duplicate uid encountered during RSE encoding: {normalized_uid}")

        record_store[normalized_uid] = {rse.Q_GRAM_ATTR: record_q_grams}
        filtered_rows.append(row)
        filtered_uids.append(normalized_uid)
        q_gram_counter.update(record_q_grams)

    return record_store, filtered_rows, filtered_uids, q_gram_counter, skipped_rows


def write_rse_q_gram_frequencies(q_gram_counter: Counter[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for q_gram, frequency in sorted(q_gram_counter.items()):
            writer.writerow([q_gram, frequency])


def count_rse_q_gram_frequencies(data: List[List[str]], q: int) -> Counter[str]:
    q_gram_counter: Counter[str] = Counter()
    for row in data:
        for value in row:
            if str(value).strip() == "":
                continue
            q_gram_counter.update(iter_rse_q_grams(value, q))
    return q_gram_counter


def count_record_store_q_grams(record_store: dict, q_gram_attr: str) -> Counter[str]:
    q_gram_counter: Counter[str] = Counter()
    for record in record_store.values():
        q_gram_counter.update(record[q_gram_attr])
    return q_gram_counter


def get_rse_all_q_grams(q: int) -> List[str]:
    return ["".join(chars) for chars in itertools.product(RSE_ALPHABET, repeat=q)]


def infer_rse_aux_source_dataset(ds_path: Path, args: argparse.Namespace) -> Path:
    if args.rse_aux_source_dataset is not None:
        return args.rse_aux_source_dataset

    candidates = [ds_path.with_name("fakename_50k.tsv")]
    if ds_path.parent.name == "noisy":
        candidates.append(ds_path.parent.parent / "fakename_50k.tsv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not infer an auxiliary RSE source dataset for {ds_path}. Pass --rse-aux-source-dataset explicitly."
    )


def resolve_rse_artifact_paths(ds_path: Path, aux_source_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    generated_dir = Path(__file__).resolve().parent / "rse-for-pprl" / "data" / "generated"
    secret_hash = md5(args.secret.encode("utf-8")).hexdigest()[:8]
    qgram_path = args.rse_qgram_frequency_file or generated_dir / f"{aux_source_path.stem}_alnum_q{args.ngram_size}_frequencies.csv"
    ref_set_path = args.rse_init_ref_set_file or generated_dir / (
        f"reference_sets_{aux_source_path.stem}_alnum_q{args.ngram_size}_k{args.rse_k}_lr{args.rse_ref_set_length}_seed{secret_hash}.csv"
    )
    return Path(ref_set_path), Path(qgram_path)


def validate_rse_reference_sets(path: Path, q: int, k: int, ref_set_length: int) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"Reference-set file not found: {path}"

    allowed_q_grams = set(get_rse_all_q_grams(q))
    q_gram_counts: Counter[str] = Counter()

    with path.open(newline="") as f:
        reader = csv.reader(f)
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            if len(row) != ref_set_length:
                return False, f"Row {row_count} has {len(row)} entries, expected {ref_set_length}"
            if len(set(row)) != len(row):
                return False, f"Row {row_count} contains duplicate q-grams"
            for q_gram in row:
                if q_gram not in allowed_q_grams:
                    return False, f"Row {row_count} contains invalid q-gram {q_gram!r}"
                q_gram_counts[q_gram] += 1

    missing = [q_gram for q_gram in allowed_q_grams if q_gram_counts[q_gram] < k]
    if missing:
        return False, f"Reference sets do not cover all alphanumeric q-grams at least {k} times"

    return True, "ok"


def ensure_rse_reference_sets(path: Path, args: argparse.Namespace) -> Path:
    is_valid, reason = validate_rse_reference_sets(path, args.ngram_size, args.rse_k, args.rse_ref_set_length)
    if is_valid and not args.overwrite:
        print(f"- Reusing RSE reference sets: {path}")
        return path

    if path.exists() and not is_valid:
        print(f"- Regenerating invalid RSE reference sets ({reason}): {path}")
    elif path.exists():
        print(f"- Regenerating RSE reference sets due to --overwrite: {path}")
    else:
        print(f"- Generating RSE reference sets: {path}")

    generator = load_rse_ref_set_generator_module()
    q_common = generator.q_gram_generator(args.ngram_size, True, True, False)
    ref_sets = generator.ref_set_generator(args.rse_ref_set_length, q_common, args.secret, args.rse_k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ref_sets)

    is_valid, reason = validate_rse_reference_sets(path, args.ngram_size, args.rse_k, args.rse_ref_set_length)
    if not is_valid:
        raise ValueError(f"Generated RSE reference sets failed validation: {reason}")

    return path


def sample_rse_auxiliary_records(
    primary_uids: List[str],
    aux_rows: List[List[str]],
    aux_uids: List[str],
    sample_size: int,
    seed: str,
) -> tuple[List[List[str]], List[str], bool]:
    if len(aux_uids) < sample_size:
        raise ValueError(
            f"Auxiliary RSE source dataset only has {len(aux_uids)} encodable rows, but {sample_size} are required."
        )

    primary_uid_set = set(primary_uids)
    candidate_indices = [idx for idx, uid in enumerate(aux_uids) if uid not in primary_uid_set]
    used_overlap = False

    if len(candidate_indices) < sample_size:
        candidate_indices = list(range(len(aux_uids)))
        used_overlap = True

    rnd = random.Random(seed)
    selected_indices = rnd.sample(candidate_indices, sample_size)
    sampled_rows = [aux_rows[idx] for idx in selected_indices]
    sampled_uids = [aux_uids[idx] for idx in selected_indices]
    return sampled_rows, sampled_uids, used_overlap


def print_csv_preview(path: Path, num_lines: int = 5) -> None:
    print("Q-Gram Frequency File Sample:")
    with path.open("r") as f:
        for _, line in zip(range(num_lines), f):
            print(f"  {line.strip()}")


def get_vah_vulnerable_q_grams(q_gram_counter: Counter[str], vuln_q_gram_count: int) -> tuple[List[str], List[str]]:
    print(f"Identifying top {vuln_q_gram_count} frequent q-grams to harden via VAH")
    sorted_q_grams = sorted(q_gram_counter.items(), key=lambda item: (item[1], item[0]))
    vuln_q_grams = [key for key, _ in sorted_q_grams[-vuln_q_gram_count:]]
    non_vuln_q_grams = [key for key, _ in sorted_q_grams[:-vuln_q_gram_count]] if vuln_q_gram_count < len(sorted_q_grams) else []
    print(f"VAH vulnerable q-grams: {vuln_q_grams}")
    print(f"VAH non-vulnerable q-gram pool size: {len(non_vuln_q_grams)}")
    return vuln_q_grams, non_vuln_q_grams


def harden_record_store_with_vah(
    record_store: dict,
    q_gram_attr: str,
    vah_instance,
    q_gram_counter: Counter[str],
    dataset_label: str,
) -> tuple[dict, Counter[str]]:
    print(f"Applying VAH hardening to {dataset_label}")
    data_dict = {uid: set(record[q_gram_attr]) for uid, record in record_store.items()}
    hardened_data_dict = vah_instance.harden_with_vah_ref_sets(copy.deepcopy(data_dict), dict(q_gram_counter))
    hardened_store = {uid: {q_gram_attr: hardened_data_dict[uid]} for uid in record_store.keys()}
    hardened_counter = count_record_store_q_grams(hardened_store, q_gram_attr)
    return hardened_store, hardened_counter


def maybe_apply_vah_hardening(
    primary_store: dict,
    primary_counter: Counter[str],
    aux_sample_store: dict,
    aux_sample_counter: Counter[str],
    aux_public_store: dict,
    aux_public_counter: Counter[str],
    args: argparse.Namespace,
    q_gram_attr: str,
) -> tuple[dict, Counter[str], dict, Counter[str]]:
    if not args.rse_hardening:
        return primary_store, primary_counter, aux_sample_store, aux_sample_counter

    if args.rse_hardening_vuln_qgrams <= 0:
        raise ValueError("--rse-hardening-vuln-qgrams must be positive when VAH hardening is enabled.")

    vah_module = load_vah_hardening_module()
    vuln_q_grams, non_vuln_q_grams = get_vah_vulnerable_q_grams(aux_public_counter, args.rse_hardening_vuln_qgrams)
    if not vuln_q_grams:
        raise ValueError("VAH hardening could not identify any vulnerable q-grams from the public auxiliary dataset.")
    if not non_vuln_q_grams:
        raise ValueError(
            "VAH hardening requires at least one non-vulnerable public q-gram. Reduce --rse-hardening-vuln-qgrams."
        )

    vah_ref_set_length = args.rse_hardening_ref_set_length or args.rse_ref_set_length
    if vah_ref_set_length <= 0:
        raise ValueError("VAH hardening reference-set length must be positive.")

    vah_instance = vah_module.VAH(args.secret, set(vuln_q_grams), non_vuln_q_grams, vah_ref_set_length)
    pub_db_q_gram_sets = {uid: set(record[q_gram_attr]) for uid, record in aux_public_store.items()}
    print("Generating VAH reference sets from the public auxiliary dataset")
    vah_instance.generate_reference_sets(pub_db_q_gram_sets)

    hardened_primary_store, hardened_primary_counter = harden_record_store_with_vah(
        primary_store,
        q_gram_attr,
        vah_instance,
        primary_counter,
        "the primary dataset",
    )
    hardened_aux_store, hardened_aux_counter = harden_record_store_with_vah(
        aux_sample_store,
        q_gram_attr,
        vah_instance,
        aux_sample_counter,
        "the sampled auxiliary dataset",
    )
    return hardened_primary_store, hardened_primary_counter, hardened_aux_store, hardened_aux_counter


def encode_with_bf(data: List[List[str]], uids: List[str], args: argparse.Namespace, diffusion=False, bf_t=10):
    from graphMatching.encoders.bf_encoder import BFEncoder

    encoder = BFEncoder(
        args.secret,
        args.bf_length,
        args.bf_bits,
        args.ngram_size,
        diffusion,
        args.bf_length,
        bf_t,
        workers=args.jobs,
    )

    if args.skip_pairs:
        # Skip pairwise similarity computation to avoid huge memory use on large datasets.
        data_joined = [[normalize_rse_value("".join(d))] for d in data]
        enc = encoder.encode(data_joined)
        enc_as_string = ["".join(map(str, bits.astype(int))) for bits in enc]
        combined = np.column_stack((data, enc_as_string, uids))
    else:
        _, combined = encoder.encode_and_compare_and_append(data, uids, metric="dice", sim=True, store_encs=False)

    return combined


def encode_with_tmh(data: List[List[str]], uids: List[str], args: argparse.Namespace):
    from graphMatching.encoders.tmh_encoder import TMHEncoder

    encoder = TMHEncoder(
        args.tmh_num_hash,
        args.tmh_hash_bits,
        args.tmh_subkeys,
        args.ngram_size,
        one_bit_hash=args.tmh_one_bit,
        random_seed=args.secret,
        verbose=args.verbose,
        workers=args.jobs,
    )

    if args.skip_pairs:
        enc = encoder.encode(data)
        enc_as_string = ["".join(map(str, bits.astype(int))) for bits in enc]
        combined = np.column_stack((data, enc_as_string, uids))
    else:
        _, combined = encoder.encode_and_compare_and_append(data, uids, metric="dice", sim=True, store_encs=False)

    return combined


def encode_with_tsh(data: List[List[str]], uids: List[str], args: argparse.Namespace):
    from graphMatching.encoders.tsh_encoder import TSHEncoder

    encoder = TSHEncoder(
        args.tsh_num_hash_func,
        args.tsh_num_hash_col,
        args.ngram_size,
        rand_mode=args.tsh_rand_mode,
        secret=args.secret,
        verbose=args.verbose,
        workers=args.jobs,
    )

    if args.skip_pairs:
        encodings = encoder.encode(data)
        combined = np.column_stack((data, encodings, uids))
    else:
        _, combined = encoder.encode_and_compare_and_append(data, uids, metric="dice", sim=True, store_encs=False)

    return combined


def encode_with_rse(data: List[List[str]], uids: List[str], args: argparse.Namespace, ds_path: Path):
    rse = load_rse_encoder_module()
    primary_store, primary_rows, primary_uids, primary_q_gram_counter, primary_skipped = build_rse_record_store(
        data,
        uids,
        args.ngram_size,
    )

    if not primary_store:
        raise ValueError(f"RSE encoding produced no encodable rows for {ds_path}")

    aux_source_path = infer_rse_aux_source_dataset(ds_path, args)
    aux_source_data, aux_source_uids, _ = read_tsv(str(aux_source_path), skip_header=False)
    aux_full_store, aux_full_rows, aux_full_uids, aux_full_q_gram_counter, aux_skipped = build_rse_record_store(
        aux_source_data,
        aux_source_uids,
        args.ngram_size,
    )

    sampled_aux_rows, sampled_aux_uids, used_overlap = sample_rse_auxiliary_records(
        primary_uids,
        aux_full_rows,
        aux_full_uids,
        len(primary_uids),
        seed=f"{args.secret}:{ds_path}:{aux_source_path}",
    )
    aux_sample_store = {uid: aux_full_store[uid] for uid in sampled_aux_uids}
    aux_sample_q_gram_counter = count_record_store_q_grams(aux_sample_store, rse.Q_GRAM_ATTR)

    primary_store, primary_q_gram_counter, aux_sample_store, aux_sample_q_gram_counter = maybe_apply_vah_hardening(
        primary_store,
        primary_q_gram_counter,
        aux_sample_store,
        aux_sample_q_gram_counter,
        aux_full_store,
        aux_full_q_gram_counter,
        args,
        rse.Q_GRAM_ATTR,
    )

    ref_set_file, q_gram_frequency_file = resolve_rse_artifact_paths(ds_path, aux_source_path, args)
    q_gram_counter = count_rse_q_gram_frequencies(aux_source_data, args.ngram_size)
    write_rse_q_gram_frequencies(q_gram_counter, q_gram_frequency_file)
    print(f"- Wrote RSE q-gram frequencies from public source dataset {aux_source_path}: {q_gram_frequency_file}")
    print_csv_preview(q_gram_frequency_file)

    ref_set_file = ensure_rse_reference_sets(ref_set_file, args)
    reference_q_gram_sets = rse.ref_set_processor.RefSetProcessor(
        str(ref_set_file),
        str(q_gram_frequency_file),
        args.rse_swap_ref_sets,
        args.secret,
    ).process_ref_q_gram_sets()

    primary_q_grams = [record[rse.Q_GRAM_ATTR] for record in primary_store.values()]
    aux_q_grams = [record[rse.Q_GRAM_ATTR] for record in aux_sample_store.values()]
    min_q_gram_length, avg_q_gram_length = rse.calculate_average_len(primary_q_grams + aux_q_grams)
    initial_signature_length = (args.rse_k + 1) * min_q_gram_length

    primary_smallest_k, primary_store = rse.gen_init_int_signature(
        primary_store,
        reference_q_gram_sets,
        initial_signature_length,
    )
    aux_smallest_k, aux_sample_store = rse.gen_init_int_signature(
        aux_sample_store,
        reference_q_gram_sets,
        initial_signature_length,
    )

    num_1_bits = min(primary_smallest_k, aux_smallest_k)
    encoded_primary = rse.extract_signatures(reference_q_gram_sets, primary_store, num_1_bits)
    _ = rse.extract_signatures(reference_q_gram_sets, aux_sample_store, num_1_bits)

    if primary_skipped:
        print(f"- RSE skipped {primary_skipped} rows from the primary dataset due to missing values or empty q-gram sets")
    if aux_skipped:
        print(f"- RSE skipped {aux_skipped} rows from the auxiliary source dataset due to missing values or empty q-gram sets")
    if used_overlap:
        print("- RSE auxiliary sample reused overlapping UIDs because the auxiliary source had too few disjoint encodable rows")

    print(f"- RSE auxiliary source dataset: {aux_source_path}")
    print(f"- RSE auxiliary sample size: {len(sampled_aux_uids)}")
    print(f"- RSE q-gram stats across both datasets: shortest={min_q_gram_length}, average={avg_q_gram_length}")
    print(f"- RSE number of 1-bits per record (shared across both datasets): {num_1_bits}")

    encodings = [encoded_primary[uid][rse.SIGNATURE_ATTR].to01() for uid in primary_uids]
    return np.column_stack((primary_rows, encodings, primary_uids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode all non-encoded datasets under data/datasets.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/datasets"), help="Directory or single TSV file to encode.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--encoders", nargs="+", choices=["bf", "tmh", "tsh", "bfd", "rse"], default=["bf", "tmh", "tsh", "bfd"],
                        help="Which encoders to run.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if encoded files already exist.")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel workers for TMH/TSH (-1 = all cores).")
    parser.add_argument("--verbose", action="store_true", help="Enable tqdm output inside TMH/TSH.")
    parser.add_argument("--secret", type=str, default=DEFAULT_SECRET, help="Secret/salt passed to the encoders.")
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE, help="N-gram size for all encoders.")
    parser.add_argument("--bf-length", type=int, default=DEFAULT_BF_LENGTH)
    parser.add_argument("--bf-bits", type=int, default=DEFAULT_BF_BITS)
    parser.add_argument("--bf-t", type=int, default=DEFAULT_BF_T)
    parser.add_argument("--bf-eld-length", type=int, default=DEFAULT_BF_ELD_LENGTH,
                        help="ELD length (used if diffusion is enabled).")
    parser.add_argument("--bf-diffusion", action="store_true", help="Enable BF diffusion.")
    parser.add_argument("--tmh-num-hash", type=int, default=DEFAULT_TMH_NUM_HASH)
    parser.add_argument("--tmh-hash-bits", type=int, default=DEFAULT_TMH_HASH_BITS)
    parser.add_argument("--tmh-subkeys", type=int, default=DEFAULT_TMH_SUBKEYS)
    parser.add_argument("--tmh-one-bit", action="store_true", default=DEFAULT_TMH_ONE_BIT,
                        help="Use 1-bit tab minhash output.")
    parser.add_argument("--tsh-num-hash-func", type=int, default=DEFAULT_TSH_NUM_HASH_FUNC)
    parser.add_argument("--tsh-num-hash-col", type=int, default=DEFAULT_TSH_NUM_HASH_COL)
    parser.add_argument("--tsh-rand-mode", choices=["PNG", "SHA"], default=DEFAULT_TSH_RAND_MODE)
    parser.add_argument("--rse-init-ref-set-file", type=Path, default=None,
                        help="Path for the RSE reference-set CSV. If omitted, a correct alphanumeric file is generated automatically.")
    parser.add_argument("--rse-qgram-frequency-file", type=Path, default=None,
                        help="Path for the generated RSE q-gram frequency CSV. If omitted, a file is generated automatically.")
    parser.add_argument("--rse-k", type=int, default=DEFAULT_RSE_K,
                        help="RSE parameter k: number of reference sets in which each q-gram must occur.")
    parser.add_argument("--rse-ref-set-length", type=int, default=DEFAULT_RSE_REF_SET_LENGTH,
                        help="Length of each RSE reference set.")
    parser.add_argument("--rse-aux-source-dataset", type=Path, default=None,
                        help="Public/auxiliary TSV dataset used to derive q-gram frequencies and sample the second dataset. Defaults to fakename_50k.tsv when available.")
    parser.add_argument("--rse-swap-ref-sets", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable the RSE frequency-based swapping step for reference sets (default: enabled).")
    parser.add_argument("--rse-hardening", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable optional VAH hardening on the RSE q-gram sets before encoding.")
    parser.add_argument("--rse-hardening-vuln-qgrams", type=int, default=10,
                        help="Number of top frequent public q-grams to harden when --rse-hardening is enabled.")
    parser.add_argument("--rse-hardening-ref-set-length", type=int, default=None,
                        help="Reference-set length for VAH hardening. Defaults to --rse-ref-set-length when omitted.")
    parser.add_argument("--skip-pairs", action="store_true", help="Skip pairwise similarity calculations to reduce memory and avoid OpenMP crashes.")
    args = parser.parse_args()

    if "rse" in args.encoders:
        if args.rse_k <= 0:
            parser.error("--rse-k must be a positive integer when '--encoders rse' is selected.")
        if args.rse_ref_set_length <= 0:
            parser.error("--rse-ref-set-length must be a positive integer when '--encoders rse' is selected.")
        if args.rse_aux_source_dataset is not None and not args.rse_aux_source_dataset.is_file():
            parser.error(f"RSE auxiliary source dataset not found: {args.rse_aux_source_dataset}")
        if args.rse_hardening_vuln_qgrams <= 0:
            parser.error("--rse-hardening-vuln-qgrams must be a positive integer.")
        if args.rse_hardening_ref_set_length is not None and args.rse_hardening_ref_set_length <= 0:
            parser.error("--rse-hardening-ref-set-length must be a positive integer when provided.")

    datasets = list(iter_plain_datasets(args.source_dir, args.recursive))
    if not datasets:
        raise SystemExit(f"No plain .tsv datasets found under {args.source_dir}")

    for ds_path in datasets:
        data, uids, header = read_tsv(str(ds_path), skip_header=False)
        print(f"\nProcessing {ds_path}")

        if "bf" in args.encoders:
            bf_out = ds_path.with_name(ds_path.stem + "_bf_encoded.tsv")
            if bf_out.exists() and not args.overwrite:
                print(f"- Skipping BF (exists): {bf_out}")
            else:
                bf_rows = encode_with_bf(data, uids, args)
                bf_header = list(header)
                bf_header.insert(-1, "bloomfilter")
                write_tsv(bf_header, bf_rows, bf_out)
                print(f"- Wrote {bf_out}")
        
        if "bfd" in args.encoders:
            bfd_out = ds_path.with_name(ds_path.stem + "_bfd_encoded.tsv")
            if bfd_out.exists() and not args.overwrite:
                print(f"- Skipping BFD (exists): {bfd_out}")
            else:
                bfd_rows = encode_with_bf(data, uids, args, True, 10)
                bfd_header = list(header)
                bfd_header.insert(-1, "bloomfilter")
                write_tsv(bfd_header, bfd_rows, bfd_out)
                print(f"- Wrote {bfd_out}")

        if "tmh" in args.encoders:
            tmh_out = ds_path.with_name(ds_path.stem + "_tmh_encoded.tsv")
            if tmh_out.exists() and not args.overwrite:
                print(f"- Skipping TMH (exists): {tmh_out}")
            else:
                tmh_rows = encode_with_tmh(data, uids, args)
                tmh_header = list(header)
                tmh_header.insert(-1, "tabminhash")
                write_tsv(tmh_header, tmh_rows, tmh_out)
                print(f"- Wrote {tmh_out}")

        if "tsh" in args.encoders:
            tsh_out = ds_path.with_name(ds_path.stem + "_tsh_encoded.tsv")
            if tsh_out.exists() and not args.overwrite:
                print(f"- Skipping TSH (exists): {tsh_out}")
            else:
                tsh_rows = encode_with_tsh(data, uids, args)
                tsh_header = list(header)
                tsh_header.insert(-1, "twostephash")
                write_tsv(tsh_header, tsh_rows, tsh_out)
                print(f"- Wrote {tsh_out}")

        if "rse" in args.encoders:
            rse_out = ds_path.with_name(ds_path.stem + "_rse_encoded.tsv")
            if rse_out.exists() and not args.overwrite:
                print(f"- Skipping RSE (exists): {rse_out}")
            else:
                rse_rows = encode_with_rse(data, uids, args, ds_path)
                rse_header = list(header)
                rse_header.insert(-1, "encoded_vector")
                write_tsv(rse_header, rse_rows, rse_out)
                print(f"- Wrote {rse_out}")


if __name__ == "__main__":
    main()
