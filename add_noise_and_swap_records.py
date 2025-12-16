#!/usr/bin/env python3
"""
Encode-first corruption pipeline: add noise to encoded TSVs while
ignoring the last two columns (encoding + uid), then swap encodings between
records. Expects files like fakename_50k_bf_encoded.tsv etc. in --input-dir.
"""
import argparse
import csv
import pathlib
import random
import re
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Sequence

def clamp(prob: float, maximum: float = 0.95) -> float:
    return max(0.0, min(prob, maximum))

def build_noise_config(level: float) -> Dict[str, float]:
    return {
        "missing_prob": clamp(0.03 * level),
        "typo_prob": clamp(0.15 * level),
        "case_prob": clamp(0.1 * level),
        "swap_name_prob": clamp(0.04 * level),
        "char_swap_prob": clamp(0.06 * level),
        "whitespace_prob": clamp(0.12 * level),
        "suffix_prob": clamp(0.05 * level),
        "uid_noise_prob": clamp(0.06 * level),
        "date_shift_prob": clamp(0.3 * level),
        "date_format_prob": clamp(0.45 * level),
        "date_text_token_prob": clamp(0.02 * level),
        "max_date_shift_days": max(1, int(12 * level)),
    }

def introduce_typo(value: str, rng: random.Random) -> str:
    if not value:
        return value
    idx = rng.randrange(len(value))
    operations = ("delete", "insert", "swap", "replace")
    op = rng.choice(operations)
    letters = "abcdefghijklmnopqrstuvwxyz"
    if op == "delete":
        return value[:idx] + value[idx + 1 :]
    if op == "insert":
        return value[:idx] + rng.choice(letters) + value[idx:]
    if op == "swap" and len(value) > 1:
        j = min(idx + 1, len(value) - 1)
        swapped = list(value)
        swapped[idx], swapped[j] = swapped[j], swapped[idx]
        return "".join(swapped)
    return value[:idx] + rng.choice(letters) + value[idx + 1 :]

def random_case(value: str, rng: random.Random) -> str:
    if not value:
        return value
    fn = rng.choice([str.lower, str.upper, str.title, str.capitalize])
    return fn(value)

def add_whitespace(value: str, rng: random.Random) -> str:
    if not value:
        return value
    prefix = " " * rng.randint(0, 2)
    suffix = " " * rng.randint(0, 2)
    return f"{prefix}{value}{suffix}"

def add_suffix(value: str, rng: random.Random) -> str:
    if not value:
        return value
    suffixes = [" Jr", " Sr", " II", " III", "-Smith"]
    return value + rng.choice(suffixes)

def swap_two_characters(value: str, rng: random.Random) -> str:
    if not value or len(value) < 2:
        return value
    i, j = rng.sample(range(len(value)), 2)
    if i == j:
        return value
    chars = list(value)
    chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)

def parse_birthday(value: str):
    formats = ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None

def format_birthday(dt: datetime, rng: random.Random, config: Dict[str, float]) -> str:
    if rng.random() < config["date_text_token_prob"]:
        return rng.choice(["unknown", "n/a", "see notes", "??"])
    if rng.random() < config["date_shift_prob"]:
        delta = rng.randint(-config["max_date_shift_days"], config["max_date_shift_days"])
        dt = dt + timedelta(days=delta)
    formats = ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d %b %Y"]
    formatted = dt.strftime(rng.choice(formats))
    if rng.random() < config["date_format_prob"]:
        formatted = re.sub(r"\b0(\d)", r"\1", formatted)
        if rng.random() < 0.3:
            formatted = formatted.replace("/", "-")
    return formatted

def mutate_name(value: str, rng: random.Random, config: Dict[str, float]) -> str:
    if rng.random() < config["missing_prob"]:
        return ""
    if rng.random() < config["char_swap_prob"]:
        value = swap_two_characters(value, rng)
    if rng.random() < config["typo_prob"]:
        value = introduce_typo(value, rng)
    if rng.random() < config["case_prob"]:
        value = random_case(value, rng)
    if rng.random() < config["whitespace_prob"]:
        value = add_whitespace(value, rng)
    if rng.random() < config["suffix_prob"]:
        value = add_suffix(value, rng)
    return value

def mutate_generic(value: str, rng: random.Random, config: Dict[str, float]) -> str:
    if rng.random() < config["missing_prob"]:
        return ""
    if rng.random() < config["char_swap_prob"]:
        value = swap_two_characters(value, rng)
    if rng.random() < config["typo_prob"]:
        value = introduce_typo(value, rng)
    if rng.random() < config["case_prob"]:
        value = random_case(value, rng)
    if rng.random() < config["whitespace_prob"]:
        value = add_whitespace(value, rng)
    return value

def mutate_encoded_row(row: Dict[str, str], fieldnames: Sequence[str], rng: random.Random, config: Dict[str, float]) -> Dict[str, str]:
    # Skip the last two columns (encoding + uid)
    mutate_fields = fieldnames[:-2] if len(fieldnames) >= 2 else []
    mutated = dict(row)
    for col in mutate_fields:
        val = row.get(col, "")
        col_lower = col.lower()
        if col in ("GivenName", "Surname"):
            mutated[col] = mutate_name(val, rng, config)
        elif "birth" in col_lower or "date" in col_lower:
            dt = parse_birthday(val)
            if dt:
                mutated[col] = format_birthday(dt, rng, config)
            elif rng.random() < config["missing_prob"]:
                mutated[col] = ""
        else:
            mutated[col] = mutate_generic(val, rng, config)
    if rng.random() < config["swap_name_prob"] and "GivenName" in mutated and "Surname" in mutated:
        mutated["GivenName"], mutated["Surname"] = mutated["Surname"], mutated["GivenName"]
    return mutated

# --- Swap helpers (from swap_encoded_rows.py) ---
def iter_encoded_files(input_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    patterns = [
        "*_bf_encoded.tsv",
        "*_bfd_encoded.tsv",
        "*_tmh_encoded.tsv",
        "*_tsh_encoded.tsv",
    ]
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            yield path

def apply_encoding_swaps(rows: List[Dict[str, str]], enc_col: str, rng: random.Random, swap_prob: float) -> None:
    if not rows or enc_col not in rows[0] or swap_prob <= 0.0:
        return
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    for k in range(0, len(indices) - 1, 2):
        if rng.random() >= swap_prob:
            continue
        i, j = indices[k], indices[k + 1]
        rows[i][enc_col], rows[j][enc_col] = rows[j][enc_col], rows[i][enc_col]

# --- Pipeline ---
def process_encoded_file(path: pathlib.Path, output_dir: pathlib.Path, rng: random.Random, config: Dict[str, float], swap_prob: float) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open(newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        rows: List[Dict[str, str]] = list(reader)
        fieldnames = reader.fieldnames or (list(rows[0].keys()) if rows else [])
    if not fieldnames:
        return path

    # Add noise (skip encoding + uid), then swap encodings.
    noisy_rows = [mutate_encoded_row(row, fieldnames, rng, config) for row in rows]
    enc_col = fieldnames[-2] if len(fieldnames) >= 2 else None
    if enc_col:
        apply_encoding_swaps(noisy_rows, enc_col, rng, swap_prob)

    output_path = output_dir / f"{path.stem}.tsv"
    with output_path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(noisy_rows)
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add noise to encoded TSVs (ignoring encoding+uid) and swap encodings between records."
    )
    parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="Directory containing *_*_encoded.tsv files.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Directory to write noisy+swapped copies.")
    parser.add_argument("--noise-level", type=float, default=1.0, help="Scales noise aggressiveness.")
    parser.add_argument("--swap-prob", type=float, default=0.01, help="Probability per random pair of rows to swap encodings.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    config = build_noise_config(args.noise_level)

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    files = list(iter_encoded_files(args.input_dir))
    if not files:
        raise SystemExit(f"No encoded TSV files found under {args.input_dir}")

    print(f"Adding noise (skip last two cols) and swapping encodings for {len(files)} files -> {args.output_dir}")
    for path in files:
        out_path = process_encoded_file(path, args.output_dir, rng, config, args.swap_prob)
        try:
            pretty = out_path.relative_to(pathlib.Path.cwd())
        except ValueError:
            pretty = out_path
        print(f"- {path.name} -> {pretty}")

if __name__ == "__main__":
    main()
