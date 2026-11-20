# Release manifest — P1 reproduction package (LOCAL PREPARATION, 2026-11-12)

Prepared under round-17 adjudication addendum demand D7 (filing-stage
preparation only). **Nothing in this manifest has been published,
pushed, or deposited.** Publication/sync of this package is a
filing-stage act under the OUTBOUND GATE (battery test 24's sync arm
stands FAIL until then: the public repository still serves the
JAIR-era package).

## Contents (this directory)

- `data/` — METR Time Horizon 1.1 runs (`runs.jsonl`, 23,235 runs)
  + `benchmark_results_1_1.yaml`, as released in METR
  eval-analysis-public @ 52cb829c7a2efb2d659285c4b1768d191d97f8d2.
- `code/` — the full analysis pipeline, incl. (added 2026-11-12)
  `p3_separation_census.py`, the stage-1 separation census demanded
  in round 17 (output: `output/p3_separation_census.json`).
- `output/` — all printed-number sources (JSON/JSONL) + figures.
- `PREREGISTRATION.md`, `README.md`, `reproduce.sh`.
- `requirements-lock.txt` — dependency versions of the environment
  that produced the current outputs (numpy 1.26.4, pandas 2.1.4,
  SciPy 1.11.4, scikit-learn 1.9.1, matplotlib 3.6.3, PyYAML 6.0.1),
  captured 2026-11-12.
- `checksums.sha256` — SHA-256 of every data file, code file and
  output file (42 entries), captured 2026-11-12 after the census
  addition.

## Check taxonomy (recompute vs compare)

- **Recompute checks** — `verify.py` (113 checks): re-derives the
  headline numbers from `data/` through `code/` and compares them
  against the values the manuscript prints. A failure means the
  pipeline no longer produces the printed number.
- **Compare checks** — `verify_printed.py` (65 checks): extracts
  numbers from the compiled PDF and compares them against the
  recomputed sources. A failure means the PDF and the pipeline
  disagree (e.g. a stale build).
- Both suites PASS on this snapshot (manuscript build 82c2f8fe,
  2026-11-20): verify.py — ALL HEADLINE NUMBERS VERIFIED (353 checks);
  verify_printed.py — ALL 522 PRINTED NUMBERS VERIFIED (0 FAIL).

## Immutability note

This snapshot is frozen as the tagged, immutable release `v1.0.0`
(2026-11-20) of the public repository, with `checksums.sha256` as its
integrity record, and archived on Figshare as version 1 (version DOI
cited in the manuscript's Code Availability statement). The bundled
METR run data is not redistributed in this snapshot — see
`data/DATA-README.md`. Any later change requires a new tag and a new
manifest.
