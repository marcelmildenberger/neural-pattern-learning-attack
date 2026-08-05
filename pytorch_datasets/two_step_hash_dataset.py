import numpy as np
import torch
from utils.pytorch_dataset_utils import *
from utils.string_utils import *
from torch.utils.data import Dataset


def parse_twostephash_string(value):
    """Parse a serialized integer set used by Two-Step Hash encodings."""
    if isinstance(value, str):
        content = value.strip("{}")
        return [int(item.strip()) for item in content.split(",")] if content else []
    return [int(item) for item in value]


def build_dataset_kwargs(*, data, column_name):
    """Build one stable vocabulary shared by train, validation, and test."""
    unique_integers = {
        value
        for encoding in data[column_name]
        for value in parse_twostephash_string(encoding)
    }
    return {"all_integers": sorted(unique_integers)}


class TwoStepHashDataset(Dataset):
    def __init__(self, data, is_labeled=False, all_integers=None, dev_mode=False, all_bi_grams=None):
        self.isLabeled = is_labeled
        self.allIntegers = all_integers
        self.integerIndices = {value: index for index, value in enumerate(all_integers)}
        self.allTwoGrams = all_bi_grams
        self.devMode = dev_mode

        self.hashTensors = data['twostephash'].apply(
            lambda row: self.hash_list_to_tensor(parse_twostephash_string(row))
        )
        self.uids = data['uid']

        if self.isLabeled:
            self.labelTensors = data.apply(lambda row: label_to_tensor(extract_bi_grams("".join(row.iloc[:-2].astype(str))), self.allTwoGrams),  axis=1)

        if dev_mode:
            self.data = data
            if self.isLabeled:
                self.data['label'] = self.data.apply(lambda row: extract_bi_grams("".join(row.iloc[:-2].astype(str))), axis=1)

    def __len__(self):
        return len(self.hashTensors)

    def __getitem__(self, idx):
        if self.isLabeled:
            return self.hashTensors[idx], self.labelTensors[idx], self.uids[idx]
        else:
            return self.hashTensors[idx], self.uids[idx]

    def hash_list_to_tensor(self, hash_list):
        hash_array = np.zeros(len(self.allIntegers), dtype=np.float32)
        for val in hash_list:
            index = self.integerIndices.get(val)
            if index is not None:
                hash_array[index] = 1
        return torch.tensor(hash_array)

