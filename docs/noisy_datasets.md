# Noise Injection Specification

This document describes the noise injection procedure and parameterization used in the evaluation of the **NEPAL attack**.  
The purpose of this noise model is to simulate common data-quality issues encountered in operational record linkage settings while maintaining full reproducibility.

The noise is applied **only to plaintext identifiers after encoding has been performed**, thereby introducing controlled inconsistencies between plaintexts and their corresponding encodings.

## Overview

Noise is injected independently into each field of a record according to fixed probabilities.  
The procedure operates at three levels:

1. **Field-level mutations** (given name, surname, date of birth)
2. **Record-level permutations** (name swapping)
3. **Encoding–plaintext mismatches** (encoding swaps)

All probabilities are fixed and identical across datasets unless stated otherwise.
## Field-Level Noise Injection

### Name Fields (Given Name, Surname)

For each name field, mutation types are evaluated **sequentially in the order listed below**.  
Once a mutation is applied, **all subsequent mutation steps for that field are skipped**.

| Mutation Type | Probability | Description |
|--------------|-------------|-------------|
| Missing value | 0.03 | Field is replaced by an empty or missing token |
| Typographical error (substitution) | 0.15 | Random character substitution |
| Case change | 0.10 | Upper/lowercase modification |
| Character swap | 0.06 | Two adjacent characters are swapped |
| Whitespace insertion | 0.12 | Random whitespace inserted |
| Name suffix addition | 0.05 | Addition of common suffixes (e.g., "Jr.", "Sr.") |


### Date Fields (Date of Birth)

Date fields are mutated independently and are subject to the following transformations:

| Mutation Type | Probability | Description |
|--------------|-------------|-------------|
| Date shift | 0.30 | Uniform random shift within ±1 to ±12 days |
| Date format modification | 0.45 | Change in formatting (e.g., `YYYY-MM-DD` → `DD/MM/YYYY`) |
| Date replaced by text token | 0.02 | Replacement with non-date token (e.g., "unknown") |

## Record-Level Perturbations

After all field-level mutations have been applied, the following record-level operations may occur:

| Operation | Probability | Description |
|----------|-------------|-------------|
| GivenName–Surname swap | 0.04 | Given name and surname are exchanged |
| Encoding swap | 0.01 | Encodings between two randomly selected records are swapped |

The encoding swap introduces **complete mismatches** between plaintext identifiers and their corresponding encodings, simulating severe labeling or integration errors.

## Design Rationale

This noise model is intended to reflect common imperfections in real-world identifiers, including:

- Typographical errors and formatting inconsistencies
- Missing or corrupted fields
- Record integration errors across heterogeneous sources

Applying noise **after encoding** intentionally creates a conservative evaluation scenario, as the attacker observes plaintexts that no longer perfectly correspond to the encoded representations.


## Reproducibility

- All mutation probabilities can be adjusted in [add_noise_and_swap_recordy.py](../add_noise_and_swap_records.py).
- Noise injection is implemented in [add_noise_and_swap_recordy.py](../add_noise_and_swap_records.py).
- Random seeds can be set to ensure exact replication of experiments.

For implementation details, refer to the corresponding source files in the repository.