# C-TAO Annotation QA

This directory contains the read-only protocol-aware QA script for the public COVTrack release. It never modifies annotation JSON.

## Install In COVTrack

Copy the **contents** of this directory into `tools/qa/` in the public COVTrack repository. The resulting public paths should be:

```text
COVTrack/tools/qa/validate_ctao_annotations.py
COVTrack/tools/qa/README.md
COVTrack/tools/qa/tests/test_validate_ctao_annotations.py
```

Do not copy the server-side `tools/qa/CTAO_QA_EXECUTION_REPORT.md` or any
server-generated `results/qa/` files into the public repository. The release
script contains no server absolute paths.

## Input JSON and command

Pass the annotation to inspect with `--ctao`. The currently known public C-TAO
location is `saved_models/ctao_dataset/ctao_base.json`; replace it with any
other released JSON path when needed. Sparse anchor, split, validation and
training-config paths are also command-line inputs. Keep placeholders until
those files are actually released.

```bash
python tools/qa/validate_ctao_annotations.py \
  --ctao saved_models/ctao_dataset/ctao_base.json \
  --original-tao <PATH_TO_SPARSE_TAO_V1_JSON> \
  --original-tao-base <PATH_TO_SPARSE_TAO_V1_BASE_JSON> \
  --category-split <PATH_TO_AUTHORITATIVE_OVMOT_SPLIT_JSON> \
  --split base-only --profile protocol-aware \
  --report results/qa/ctao_qa_report.json --sha256
```

From the COVTrack repository root, create `results/qa/` if necessary. If the
JSON is not declared `base-only`, omit `--split base-only` and the category
split. For a base-only audit, provide an authoritative `--category-split`;
without it the leakage check fails rather than assuming a split.

The default profile returns only `PASS` or `FAIL`. It validates schema/references, effective bbox validity, duplicates, track/category identity, semantic sparse anchors, and effective base-only leakage. Image-ID remapping, raw JSON order, allowed subset filtering, dataloader-skipped records, tolerated boundary overshoot, and reference metadata fallback do not create warnings. Use `--verbose` to retain tolerated bbox/fallback events as INFO in JSON.

If raw novel labels exist, provide `--training-config <CONFIG.py>` and `--dataloader-source <SOURCE.py>` to reproduce the effective training filter. Raw novel count zero does not require a matching config filename.

The default terminal output is intentionally compact:

```text
C-TAO annotation QA
Schema and references: PASS
Bounding-box validity: PASS
Track/category consistency: PASS
Sparse-anchor preservation: PASS
Base-only training protocol: PASS
Final status: PASS
```

Temporal screening is disabled by default. Enable non-blocking diagnostics with:

```bash
python tools/qa/validate_ctao_annotations.py <THE_ARGUMENTS_ABOVE> \
  --temporal-diagnostics --temporal-flags ctao_temporal_diagnostics.csv
```

Temporal events remain diagnostics in JSON/CSV and never affect `PASS` or the exit code. `--profile strict-raw` retains the forensic raw-file checks and requires `--temporal-flags`.

Exit code is 0 for PASS, 1 for FAIL, and 2 for execution/configuration failure. Python 3.8+ standard library only.

`PASS` means all requested protocol-critical checks passed. `FAIL` means at
least one structural/reference, effective bbox, identity/category, unexplained
anchor, duplicate, or effective base-only leakage check failed. Diagnostic
events never change the status. The JSON report is UTF-8 and records selected
input paths, sizes, optional SHA256, counts, anchor/leakage summaries, hard
errors and optional diagnostics. The CSV is written only when
`--temporal-diagnostics` is enabled.

```bash
python -m py_compile tools/qa/validate_ctao_annotations.py
python -m unittest discover -s tools/qa/tests -v
```
