"""
Batch-encode all non-encoded TSV datasets under data/datasets (and subfolders) using
the same encoder settings used during the GMA runs. Produces *_bf_encoded.tsv,
*_bfd_encoded.tsv, *_tmh_encoded.tsv and *_tsh_encoded.tsv files with the encoder
column inserted right before the uid column.

Example:
python encode_datasets.py --source-dir data/datasets --recursive
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from utils.encoder_registry import encoded_dataset_suffixes, get_encoder_spec
from utils.data_pipeline import read_tsv


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


@dataclass(frozen=True)
class EncodingJob:
    alias: str
    label: str
    spec_alias: str
    encode: Callable[[List[List[str]], List[str], argparse.Namespace], Sequence[Sequence]]
    diffuse: bool = False

    def output_path(self, dataset_path: Path) -> Path:
        spec = get_encoder_spec(self.spec_alias)
        return Path(spec.encoded_path(str(dataset_path), diffuse=self.diffuse))

    def output_header(self, source_header: Sequence[str]) -> list[str]:
        spec = get_encoder_spec(self.spec_alias)
        header = list(source_header)
        header.insert(-1, spec.column_name)
        return header


def iter_plain_datasets(root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.tsv" if recursive else "*.tsv"
    encoded_suffixes = encoded_dataset_suffixes()
    for path in root.glob(pattern):
        name = path.name
        if name.endswith(encoded_suffixes):
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
    _, combined = encoder.encode_and_compare_and_append(data, uids, metric="dice", sim=True, store_encs=False)
    return combined


def encode_with_bfd(data: List[List[str]], uids: List[str], args: argparse.Namespace):
    return encode_with_bf(data, uids, args, diffusion=True, bf_t=args.bf_t)


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
    _, combined = encoder.encode_and_compare_and_append(data, uids, metric="dice", sim=True, store_encs=False)
    return combined


ENCODING_JOBS = {
    "bf": EncodingJob("bf", "BF", "bf", encode_with_bf),
    "bfd": EncodingJob("bfd", "BFD", "bf", encode_with_bfd, diffuse=True),
    "tmh": EncodingJob("tmh", "TMH", "tmh", encode_with_tmh),
    "tsh": EncodingJob("tsh", "TSH", "tsh", encode_with_tsh),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode all non-encoded datasets under data/datasets.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/datasets"), help="Where to look for .tsv files.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--encoders", nargs="+", choices=sorted(ENCODING_JOBS), default=["bf", "tmh", "tsh", "bfd"],
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
    args = parser.parse_args()

    datasets = list(iter_plain_datasets(args.source_dir, args.recursive))
    if not datasets:
        raise SystemExit(f"No plain .tsv datasets found under {args.source_dir}")

    for ds_path in datasets:
        data, uids, header = read_tsv(str(ds_path), skip_header=False)
        print(f"\nProcessing {ds_path}")

        for alias in args.encoders:
            job = ENCODING_JOBS[alias]
            out_path = job.output_path(ds_path)
            if out_path.exists() and not args.overwrite:
                print(f"- Skipping {job.label} (exists): {out_path}")
                continue
            rows = job.encode(data, uids, args)
            write_tsv(job.output_header(header), rows, out_path)
            print(f"- Wrote {out_path}")


if __name__ == "__main__":
    main()
