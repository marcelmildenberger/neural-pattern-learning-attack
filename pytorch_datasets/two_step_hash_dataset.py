import numpy as np
import torch
from nepal.utils.pytorch_dataset_utils import *
from nepal.utils.string_utils import *
from torch.utils.data import Dataset


class TwoStepHashDataset(Dataset):
    def __init__(self, data, is_labeled=False, all_integers=None, dev_mode=False, all_bi_grams=None):
        self.isLabeled = is_labeled
        self.allIntegers = all_integers
        self.allTwoGrams = all_bi_grams
        self.devMode = dev_mode

        self.hashTensors = data['twostephash'].apply(lambda row: self.hash_list_to_tensor(self.parse_twostephash_string(row)))
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

    def parse_twostephash_string(self, twostephash_str):
        """Parse the string representation of a set to extract actual integers."""
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

    def hash_list_to_tensor(self, hash_list):
        hash_array = np.zeros(len(self.allIntegers), dtype=np.float32)
        for val in hash_list:
            if val in self.allIntegers:
                index = self.allIntegers.index(val)
                hash_array[index] = 1
        return torch.tensor(hash_array)


