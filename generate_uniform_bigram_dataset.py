"""Generate a bigram-balanced fake-name TSV.

This script takes an existing TSV file with the columns ``GivenName``,
``Surname``, ``Birthday`` and ``uid`` and writes a new TSV with the same
shape, but with randomized values drawn from a de Bruijn sequence so that
every possible character bigram over the file's alphabet appears with (almost)
equal frequency across the whole output. The ``uid`` column is left untouched
to preserve record identities.

Usage:
    python generate_uniform_bigram_dataset.py input.tsv \
        --output datasets/fakename_uniform.tsv --seed 7

The output keeps the original row count and field lengths. Because we stream
characters from a de Bruijn cycle, the maximum imbalance between any two
bigrams is at most one occurrence when the total character count is not an
exact multiple of |alphabet|^2.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List


def de_bruijn(alphabet: List[str], order: int = 2) -> List[str]:
    """Return a de Bruijn sequence of ``alphabet`` for bigrams (order=2).

    The returned list is cyclic: iterating over it in a loop guarantees that
    every possible length-``order`` substring over ``alphabet`` appears
    exactly once per cycle.
    """

    k = len(alphabet)
    if k < 2:
        raise ValueError("Alphabet must contain at least two symbols")

    a = [0] * (k * order)
    sequence: List[int] = []

    def db(t: int, p: int) -> None:
        if t > order:
            if order % p == 0:
                sequence.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return [alphabet[i] for i in sequence]


class BigramStream:
    """Cyclic character stream with (almost) uniform bigram coverage."""

    def __init__(self, alphabet: Iterable[str], seed: int | None = None):
        rng = random.Random(seed)
        self.alphabet = list(alphabet)
        rng.shuffle(self.alphabet)
        self.sequence = de_bruijn(self.alphabet, order=2)
        self.pos = rng.randrange(len(self.sequence))

    def take(self, length: int) -> str:
        seq_len = len(self.sequence)
        out: List[str] = []
        for _ in range(length):
            out.append(self.sequence[self.pos])
            self.pos += 1
            if self.pos == seq_len:
                self.pos = 0
        return "".join(out)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError("Input file is missing a header row")
    return rows


def build_alphabet(
    rows: List[Dict[str, str]], exclude_fields: Iterable[str] | None = None
) -> List[str]:
    chars = set()
    exclude = {field.lower() for field in exclude_fields} if exclude_fields else set()
    for row in rows:
        for key, value in row.items():
            if key.lower() in exclude:
                continue
            chars.update(value)
    if len(chars) < 2:
        raise ValueError("Need at least two distinct characters to form bigrams")
    return sorted(chars)


def generate_uniform_bigram_dataset(
    rows: List[Dict[str, str]], seed: int | None = None
) -> List[Dict[str, str]]:
    """Return rows with randomized values preserving field lengths.

    Characters are drawn sequentially from a single BigramStream to keep
    bigram frequencies uniform across *all* fields and rows combined.
    """

    if not rows:
        return []

    frozen_fields = {"uid"}

    alphabet = build_alphabet(rows, exclude_fields=frozen_fields)
    stream = BigramStream(alphabet, seed=seed)

    fieldnames = list(rows[0].keys())
    generated: List[Dict[str, str]] = []

    for row in rows:
        new_row: Dict[str, str] = {}
        for col in fieldnames:
            if col.lower() in frozen_fields:
                new_row[col] = row[col]
                continue
            length = len(row[col])
            new_row[col] = stream.take(length)
        generated.append(new_row)

    return generated


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input TSV with fake-name data")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to write the randomized TSV (default: <input>_uniform.tsv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: system entropy)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = (
        args.output
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_uniform.tsv")
    )

    rows = read_rows(input_path)
    if not rows:
        raise SystemExit("Input TSV is empty")

    randomized_rows = generate_uniform_bigram_dataset(rows, seed=args.seed)
    write_rows(output_path, list(rows[0].keys()), randomized_rows)

    print(f"Wrote {len(randomized_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
