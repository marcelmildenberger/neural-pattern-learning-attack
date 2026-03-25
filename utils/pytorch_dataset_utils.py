import numpy as np
import torch


def normalized_uid_column_name(column_name):
    return "uid" if str(column_name).strip().lower() in {"uid", "id"} else column_name


def normalize_uid_dataframe_columns(df):
    renamed = {
        column_name: normalized_uid_column_name(column_name)
        for column_name in df.columns
    }
    if renamed == {column_name: column_name for column_name in df.columns}:
        return df
    return df.rename(columns=renamed)


def get_uid_series(df):
    normalized = normalize_uid_dataframe_columns(df)
    return normalized["uid"], normalized


def label_to_tensor(label, allTwoGrams):
        label_vector = np.zeros(len(allTwoGrams), dtype=np.float32)
        for gram in label:
            if gram in allTwoGrams:
                index = allTwoGrams.index(gram)
                label_vector[index] = 1
        return torch.tensor(label_vector)

def bit_string_to_tensor(bit_string):
    bit_string_array = np.array([int(bit) for bit in bit_string], dtype=np.float32)
    return torch.tensor(bit_string_array)
