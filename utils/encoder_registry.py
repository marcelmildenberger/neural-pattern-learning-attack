from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    cli_alias: str
    encoded_suffix: str
    column_name: str
    dataset_class: Optional[str]
    diffused_suffix: Optional[str] = None
    needs_integer_vocabulary: bool = False

    def encoded_path(self, data_path: str, diffuse: bool = False) -> str:
        suffix = self.diffused_suffix if diffuse and self.diffused_suffix else self.encoded_suffix
        path = Path(data_path)
        return str(path.with_name(path.stem + suffix))

    def load_dataset_class(self):
        if not self.dataset_class:
            raise ValueError(
                f"No PyTorch dataset class is registered for encoder '{self.name}'. "
                "Add one before running NEPAL training with this encoder."
            )
        module_name, class_name = self.dataset_class.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, class_name)


ENCODER_SPECS = {
    "BloomFilter": EncoderSpec(
        name="BloomFilter",
        cli_alias="bf",
        encoded_suffix="_bf_encoded.tsv",
        diffused_suffix="_bfd_encoded.tsv",
        column_name="bloomfilter",
        dataset_class="pytorch_datasets.bloom_filter_dataset.BloomFilterDataset",
    ),
    "TabMinHash": EncoderSpec(
        name="TabMinHash",
        cli_alias="tmh",
        encoded_suffix="_tmh_encoded.tsv",
        column_name="tabminhash",
        dataset_class="pytorch_datasets.tab_min_hash_dataset.TabMinHashDataset",
    ),
    "TwoStepHash": EncoderSpec(
        name="TwoStepHash",
        cli_alias="tsh",
        encoded_suffix="_tsh_encoded.tsv",
        column_name="twostephash",
        dataset_class="pytorch_datasets.two_step_hash_dataset.TwoStepHashDataset",
        needs_integer_vocabulary=True,
    ),
    "RSE": EncoderSpec(
        name="RSE",
        cli_alias="rse",
        encoded_suffix="_rse_encoded.tsv",
        column_name="rse",
        dataset_class=None,
    ),
}

ENCODER_ALIASES = {
    spec.cli_alias: name
    for name, spec in ENCODER_SPECS.items()
}


def get_encoder_spec(name_or_alias: str) -> EncoderSpec:
    name = ENCODER_ALIASES.get(name_or_alias, name_or_alias)
    try:
        return ENCODER_SPECS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(list(ENCODER_SPECS) + list(ENCODER_ALIASES)))
        raise ValueError(f"Unsupported encoder '{name_or_alias}'. Supported encoders: {supported}") from exc


def supported_nepal_encoders() -> list[str]:
    return [name for name, spec in ENCODER_SPECS.items() if spec.dataset_class]


def supported_encoder_aliases(include_without_dataset: bool = False) -> list[str]:
    return [
        spec.cli_alias
        for spec in ENCODER_SPECS.values()
        if include_without_dataset or spec.dataset_class
    ]


def encoded_dataset_path(data_path: str, algo: str, diffuse: bool = False) -> str:
    return get_encoder_spec(algo).encoded_path(data_path, diffuse=diffuse)


def encoded_dataset_suffixes(include_diffused: bool = True) -> tuple[str, ...]:
    suffixes = []
    for spec in ENCODER_SPECS.values():
        suffixes.append(spec.encoded_suffix)
        if include_diffused and spec.diffused_suffix:
            suffixes.append(spec.diffused_suffix)
    return tuple(suffixes)
