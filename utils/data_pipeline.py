import csv
from typing import Sequence

from utils.encoder_registry import encoded_dataset_path


def read_tsv(path: str, skip_header: bool = True, as_dict: bool = False, delim: str = "\t") -> Sequence[Sequence[str]]:
    data = {} if as_dict else []
    uids = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)
        for row in reader:
            if as_dict:
                assert len(row) == 3, "Dict mode only supports rows with two values + uid"
                data[row[0]] = row[1]
            else:
                data.append(row[:-1])
                uids.append(row[-1])
    return data, uids, header


def save_tsv(data, path: str, delim: str = "\t") -> None:
    with open(path, "w", newline="") as f:
        csvwriter = csv.writer(f, delimiter=delim)
        csvwriter.writerows(data)


def resolve_encoded_dataset_path(data_path: str, algo: str, diffuse: bool = False) -> str:
    return encoded_dataset_path(data_path, algo, diffuse=diffuse)
