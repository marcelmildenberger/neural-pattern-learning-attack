from utils.pytorch_dataset_utils import *
from utils.string_utils import *
from torch.utils.data import Dataset


class RoundBasedEncodingSchemeDataset(Dataset):
    def __init__(self, data, is_labeled=False, all_bi_grams=None, dev_mode=False):
        self.isLabeled = is_labeled
        self.allTwoGrams = all_bi_grams
        self.devMode = dev_mode
        
        self.bitStringTensors = data["encoded_vector"].apply(
            lambda row: bit_string_to_tensor(list(str(row)))
        )
        self.uids = data["uid"]

        if self.isLabeled:
            self.labelTensors = data.apply(
                lambda row: label_to_tensor(
                    extract_bi_grams("".join(row.iloc[:-2].astype(str))), self.allTwoGrams
                ),
                axis=1,
            )

        if dev_mode:
            self.data = data
            if self.isLabeled:
                self.data["label"] = self.data.apply(
                    lambda row: extract_bi_grams("".join(row.iloc[:-2].astype(str))),
                    axis=1,
                )

    def __len__(self):
        return len(self.bitStringTensors)

    def __getitem__(self, idx):
        if self.isLabeled:
            return self.bitStringTensors[idx], self.labelTensors[idx], self.uids[idx]
        else:
            return self.bitStringTensors[idx], self.uids[idx]
