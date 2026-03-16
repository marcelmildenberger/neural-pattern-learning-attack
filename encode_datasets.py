"""
Batch-encode all non-encoded TSV datasets under data/datasets (and subfolders) using
the same encoder settings used during the GMA runs. Produces *_bf_encoded.tsv,
*_tmh_encoded.tsv, *_tsh_encoded.tsv and *_rse_encoded.tsv files with the
encoder column inserted right before the uid column.

Example:
python encode_datasets.py --source-dir data/datasets --recursive
"""

import argparse
import csv
import importlib.util
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from graphMatching.encoders.bf_encoder import BFEncoder
from graphMatching.encoders.tmh_encoder import TMHEncoder
from graphMatching.encoders.tsh_encoder import TSHEncoder
from utils.utils import read_tsv


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


def iter_plain_datasets(root: Path, recursive: bool) -> Iterable[Path]:
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


def normalize_rse_value(value: str) -> str:
    return str(value).strip().lower().replace(" ", "")


def extract_rse_q_grams(value: str, q: int) -> set[str]:
    normalized = normalize_rse_value(value)
    if len(normalized) < q:
        return set()
    return {normalized[i:i + q] for i in range(len(normalized) - (q - 1))}


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
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for q_gram, frequency in sorted(q_gram_counter.items()):
            writer.writerow([q_gram, frequency])


def encode_with_bf(data: List[List[str]], uids: List[str], args: argparse.Namespace, diffusion=False, bf_t=10):
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
        data_joined = [["".join(d).lower()] for d in data]
        enc = encoder.encode(data_joined)
        enc_as_string = ["".join(map(str, bits.astype(int))) for bits in enc]
        combined = np.column_stack((data, enc_as_string, uids))
    else:
        _, combined = encoder.encode_and_compare_and_append(data, uids, metric="dice", sim=True, store_encs=False)

    return combined


def encode_with_tmh(data: List[List[str]], uids: List[str], args: argparse.Namespace):
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
    record_store, filtered_rows, filtered_uids, q_gram_counter, skipped_rows = build_rse_record_store(
        data,
        uids,
        args.ngram_size,
    )

    if not record_store:
        raise ValueError(f"RSE encoding produced no encodable rows for {ds_path}")

    temp_dir = None
    try:
        if args.rse_qgram_frequency_file is not None:
            q_gram_frequency_file = args.rse_qgram_frequency_file
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="rse_qgrams_")
            q_gram_frequency_file = Path(temp_dir.name) / f"{ds_path.stem}_qgram_frequencies.csv"
            write_rse_q_gram_frequencies(q_gram_counter, q_gram_frequency_file)
            print(f"- Derived RSE q-gram frequencies from dataset: {q_gram_frequency_file}")

        reference_q_gram_sets = rse.ref_set_processor.RefSetProcessor(
            str(args.rse_init_ref_set_file),
            str(q_gram_frequency_file),
            args.rse_swap_ref_sets,
            args.secret,
        ).process_ref_q_gram_sets()

        min_q_gram_length, avg_q_gram_length = rse.calculate_average_len(
            [record[rse.Q_GRAM_ATTR] for record in record_store.values()]
        )
        initial_signature_length = (args.rse_k + 1) * min_q_gram_length
        min_one_bits, record_store = rse.gen_init_int_signature(
            record_store,
            reference_q_gram_sets,
            initial_signature_length,
        )
        encoded_records = rse.extract_signatures(reference_q_gram_sets, record_store, min_one_bits)

        if skipped_rows:
            print(f"- RSE skipped {skipped_rows} rows with missing values or empty q-gram sets")
        print(f"- RSE q-gram stats: shortest={min_q_gram_length}, average={avg_q_gram_length}")
        print(f"- RSE number of 1-bits per record: {min_one_bits}")

        encodings = [encoded_records[uid][rse.SIGNATURE_ATTR].to01() for uid in filtered_uids]
        combined = np.column_stack((filtered_rows, encodings, filtered_uids))
        return combined
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode all non-encoded datasets under data/datasets.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/datasets"), help="Where to look for .tsv files.")
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
                        help="CSV file with the initial reference sets required by the RSE encoder.")
    parser.add_argument("--rse-qgram-frequency-file", type=Path, default=None,
                        help="CSV file with q-gram frequencies for the RSE encoder. Defaults to deriving frequencies from each dataset.")
    parser.add_argument("--rse-k", type=int, default=None,
                        help="RSE parameter k: number of reference sets in which each q-gram must occur.")
    parser.add_argument("--rse-swap-ref-sets", action="store_true",
                        help="Enable the RSE frequency-based swapping step for reference sets.")
    parser.add_argument("--skip-pairs", action="store_true", help="Skip pairwise similarity calculations to reduce memory and avoid OpenMP crashes.")
    args = parser.parse_args()

    if "rse" in args.encoders:
        if args.rse_init_ref_set_file is None:
            parser.error("--rse-init-ref-set-file is required when '--encoders rse' is selected.")
        if not args.rse_init_ref_set_file.is_file():
            parser.error(f"RSE reference set file not found: {args.rse_init_ref_set_file}")
        if args.rse_qgram_frequency_file is not None and not args.rse_qgram_frequency_file.is_file():
            parser.error(f"RSE q-gram frequency file not found: {args.rse_qgram_frequency_file}")
        if args.rse_k is None or args.rse_k <= 0:
            parser.error("--rse-k must be a positive integer when '--encoders rse' is selected.")

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
