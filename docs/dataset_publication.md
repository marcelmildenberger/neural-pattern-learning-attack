# Dataset Publication Strategy

The repository keeps a compact, immediately runnable artifact bundle under `data/datasets`:

- plaintext FakeName subsets used by the experiment matrix;
- their noisy BF, TMH, and TSH encodings;
- the Titanic example and clean example encodings.

Large or third-party source datasets are not duplicated when redistribution rights or repository size make that inappropriate. Obtain those datasets from their cited source, normalize them to the TSV layout in the main README, and generate encodings with `encode_datasets.py`.

When publishing an additional bundle, retain the plaintext dataset stem, use suffixes from `utils/encoder_registry.py`, and publish SHA-256 checksums alongside the bundle.
