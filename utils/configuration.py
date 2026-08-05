"""Lightweight normalization and preflight validation for NEPAL configs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from utils.encoder_registry import get_encoder_spec, supported_nepal_encoders


REQUIRED_SECTIONS = (
    "GLOBAL_CONFIG",
    "ENC_CONFIG",
    "EMB_CONFIG",
    "ALIGN_CONFIG",
    "NEPAL_CONFIG",
)


def _canonical_encoder(value: Any, *, allow_none: bool) -> str | None:
    if value is None or value == "None":
        if allow_none:
            return value
        raise ValueError("AliceAlgo must name an encoder with a NEPAL dataset loader.")
    return get_encoder_spec(str(value)).name


def _required_inputs(config: dict) -> list[tuple[str, str | None]]:
    global_config = config["GLOBAL_CONFIG"]
    encoder_config = config["ENC_CONFIG"]
    data_path = str(global_config["Data"])
    graph_matching = bool(global_config.get("GraphMatchingAttack", False))

    parties = ("Alice", "Eve") if graph_matching else ("Alice",)
    inputs = []
    for party in parties:
        algo = encoder_config.get(f"{party}Algo")
        if algo is None or algo == "None":
            inputs.append((data_path, None))
            continue
        spec = get_encoder_spec(algo)
        if graph_matching and not spec.precomputed_encoder_class:
            raise ValueError(
                f"Encoder '{spec.name}' has no GMA adapter for {party}. "
                "Register precomputed_encoder_class or disable GraphMatchingAttack."
            )
        inputs.append(
            (
                spec.encoded_path(
                    data_path,
                    diffuse=bool(encoder_config.get(f"{party}Diffuse", False)),
                ),
                spec.column_name,
            )
        )
    return list(dict.fromkeys(inputs))


def required_input_paths(config: dict) -> list[str]:
    return list(dict.fromkeys(path for path, _ in _required_inputs(config)))


def normalize_and_validate_config(config: dict, *, check_files: bool = True) -> dict:
    """Return a normalized copy or raise a user-facing ``ValueError``."""
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        raise ValueError("Missing configuration sections: " + ", ".join(missing_sections))

    normalized = copy.deepcopy(config)
    global_config = normalized["GLOBAL_CONFIG"]
    encoder_config = normalized["ENC_CONFIG"]
    nepal_config = normalized["NEPAL_CONFIG"]

    global_config.setdefault("Seed", 42)
    global_config.setdefault("DeterministicAlgorithms", True)
    nepal_config.setdefault("EarlyStopThreshold", 0.99)

    encoder_config["AliceAlgo"] = _canonical_encoder(
        encoder_config.get("AliceAlgo"),
        allow_none=False,
    )
    encoder_config["EveAlgo"] = _canonical_encoder(
        encoder_config.get("EveAlgo"),
        allow_none=True,
    )

    if encoder_config["AliceAlgo"] not in supported_nepal_encoders():
        raise ValueError(
            f"Encoder '{encoder_config['AliceAlgo']}' has no NEPAL dataset loader. "
            "Register dataset_class before using it as AliceAlgo."
        )

    overlap = float(global_config.get("Overlap", 0))
    if not 0 < overlap <= 1:
        raise ValueError("GLOBAL_CONFIG.Overlap must be greater than 0 and at most 1.")

    train_size = float(nepal_config.get("TrainSize", 0))
    if not 0 < train_size < 1:
        raise ValueError("NEPAL_CONFIG.TrainSize must be greater than 0 and less than 1.")

    for key in ("ParallelTrials", "NumSamples", "Epochs"):
        if int(nepal_config.get(key, 0)) < 1:
            raise ValueError(f"NEPAL_CONFIG.{key} must be at least 1.")

    if check_files:
        required_inputs = _required_inputs(normalized)
        missing_paths = [path for path, _ in required_inputs if not Path(path).is_file()]
        if missing_paths:
            formatted = "\n".join(f"- {path}" for path in missing_paths)
            raise ValueError("Missing experiment input files:\n" + formatted)

        for input_path, encoding_column in required_inputs:
            with Path(input_path).open("r", encoding="utf-8") as handle:
                header = handle.readline().rstrip("\r\n").split("\t")
            if not header or header[-1] != "uid":
                raise ValueError(f"Input must end with a 'uid' column: {input_path}")
            if encoding_column and (len(header) < 2 or header[-2] != encoding_column):
                raise ValueError(
                    f"Expected '{encoding_column}' immediately before 'uid' in {input_path}."
                )

    return normalized
