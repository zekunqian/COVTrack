# C-TAO Annotation QA

This directory provides a read-only quality-assurance tool for C-TAO and TAO COCO-video annotation files. The tool never edits, crops, deletes, or rewrites annotation JSON.

## Installation

Place these files in the COVTrack repository as:

```text
tools/qa/validate_ctao_annotations.py
tools/qa/README.md
tools/qa/tests/test_validate_ctao_annotations.py
```

The validator uses only the Python standard library.

## Basic Usage

Run the command from the COVTrack repository root. Supply the annotation to inspect with `--ctao`; the filename is not fixed. The currently documented COVTrack annotation location is:

```text
saved_models/ctao_dataset/ctao_base.json
```

For a base-only audit, also supply an authoritative category split. Sparse TAO reference files are required for anchor-preservation checks; use explicit paths for files that are distributed separately.

```bash
mkdir -p results/qa

python tools/qa/validate_ctao_annotations.py \
  --ctao saved_models/ctao_dataset/ctao_base.json \
  --original-tao <PATH_TO_SPARSE_TAO_JSON> \
  --original-tao-base <PATH_TO_BASE_SPARSE_TAO_JSON> \
  --category-split <PATH_TO_AUTHORITATIVE_CATEGORY_SPLIT_JSON> \
  --split base-only \
  --profile protocol-aware \
  --report results/qa/ctao_qa_report.json \
  --sha256
```

If the annotation is not a base-only file, omit `--split base-only` and `--category-split`. If raw novel labels are present in a declared base-only JSON, provide the training configuration and dataloader source so the effective category filter can be reproduced:

```bash
  --training-config <PATH_TO_TRAINING_CONFIG.py> \
  --dataloader-source <PATH_TO_DATALOADER_SOURCE.py>
```

The validator does not assume that placeholder files exist. A missing or unrecognized category split prevents the base-only leakage check instead of silently passing it.

## Default Output

The default `protocol-aware` profile performs only protocol-critical checks and returns `PASS` or `FAIL`:

```text
C-TAO annotation QA
Schema and references: PASS
Bounding-box validity: PASS
Track/category consistency: PASS
Sparse-anchor preservation: PASS
Base-only training protocol: PASS
Final status: PASS
```

`PASS` means all requested hard checks passed. `FAIL` means at least one hard error was found, such as an invalid reference, duplicate annotation, newly added invalid bbox that reaches training, changed track/category identity, unexplained anchor loss, or effective novel-label leakage in a base-only training protocol.

The JSON report is UTF-8 and includes dataset counts, selected input paths, optional SHA256 values, anchor-preservation statistics, raw/effective category leakage counts, hard-error summaries, and bounded examples. The default command does not create a temporal CSV.

## Optional Temporal Diagnostics

Temporal checks are disabled by default. They can be enabled explicitly for manual review:

```bash
python tools/qa/validate_ctao_annotations.py \
  --ctao saved_models/ctao_dataset/ctao_base.json \
  --split base-only \
  --category-split <PATH_TO_AUTHORITATIVE_CATEGORY_SPLIT_JSON> \
  --report results/qa/ctao_qa_report.json \
  --temporal-diagnostics \
  --temporal-flags results/qa/ctao_temporal_diagnostics.csv
```

Temporal events include frame gaps, low adjacent IoU, re-entry, center displacement, box-scale changes, aspect-ratio changes, repeated identical boxes, and short tracks. They are reported as diagnostics only; they never become warnings or errors, never change `PASS`, and never delete annotations.

Use `--verbose` to record tolerated bbox boundary overshoot and reference metadata fallback as INFO in the JSON report. These normal construction/reference details are silent by default.

## Strict Profile

`--profile strict-raw` is available for forensic comparison with the raw JSON file. It retains stricter checks for raw image/annotation IDs, JSON annotation order, inherited bbox defects, raw novel labels, and temporal diagnostics. Use `--temporal-flags` with strict-raw.

## Exit Codes and Tests

```text
0  PASS
1  FAIL
2  command/configuration error
```

Run the synthetic regression tests from the repository root:

```bash
python -m py_compile tools/qa/validate_ctao_annotations.py
python -m unittest discover -s tools/qa/tests -v
```

The script is read-only. It does not determine whether a box is sufficiently tight, whether an occluded object should be labeled, whether a difficult identity decision is correct, whether model output was used during annotation, or whether human adjudication was completed.
