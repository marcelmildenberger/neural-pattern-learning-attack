# Adding Encoders

NEPAL has two separate encoder-related layers:

1. Dataset encoding: creates `*_encoded.tsv` files from plaintext records.
2. NEPAL training input: converts an encoded column into a fixed-size PyTorch tensor and pairs it with plaintext q-gram labels.

Most new encoders need changes in both layers. If an encoder output uses the same on-disk format as an existing encoder, the existing dataset loader may be reused.

## Files to Update

### 1. Register Encoder Metadata

Add an entry to `utils/encoder_registry.py`.

Each `EncoderSpec` declares:

- `name`: value used in `ENC_CONFIG["AliceAlgo"]`
- `cli_alias`: short name for scripts such as `bf`, `tmh`, or `tsh`
- `encoded_suffix`: file suffix such as `_bf_encoded.tsv`
- `column_name`: encoded TSV column immediately before `uid`
- `dataset_class`: PyTorch dataset class used by NEPAL
- `needs_integer_vocabulary`: set this when a sparse set encoding needs a stable integer-to-index vocabulary

### 2. Add a PyTorch Dataset Loader

Create `pytorch_datasets/<encoder>_dataset.py` when the encoded representation needs new tensorization.

The loader must return:

```python
encoding_tensor, label_tensor, uid
```

The label tensor should be produced from plaintext columns using the same convention as the existing loaders:

```python
label_to_tensor(extract_bi_grams("".join(row.iloc[:-2].astype(str))), all_bi_grams)
```

This assumes the encoded TSV layout is:

```text
plaintext columns ...    encoded_column    uid
```

### 3. Add Dataset Encoding Support

If the repository should generate encoded TSVs for the new encoder, add a constructor and CLI branch in `encode_datasets.py`.

The output must insert the encoded column immediately before `uid`, and the column name must match `EncoderSpec.column_name`.

### 4. Consider GMA Mode

Synthetic NEPAL mode only needs the encoded TSV and PyTorch dataset loader. GMA-NEPAL mode also needs graph-matching support.

If the new encoder is used with `GraphMatchingAttack=true`, `graphMatching/gma_pipeline.py` needs a compatible precomputed encoder path. Existing supported shapes are:

- binary bit strings: `PrecomputedBinaryEncoder`
- integer sets: `PrecomputedSetEncoder`
- TabMinHash strings: `PrecomputedTMHEncoder`

### 5. Update Configs and Docs

Add the encoder name to relevant experiment configs and docs only after the loader and encoded file naming are in place.

## Things to Consider

- Tensor dimensions must be stable across train, validation, and test splits.
- Label generation currently targets 2-grams over lowercase letters and digits. If a new encoder relies on another alphabet or q-gram length, update `utils/string_utils.py` and the paper-facing docs accordingly.
- Normalization must be consistent between encoding generation and label generation.
- Encoded data must include stable unique `uid` values.
- If you change loader semantics, clear `data/cache/` before rerunning experiments.
- If the encoder output is variable length, define a deterministic vectorization strategy before training.
