# Bundled data — not redistributed in this archive

The locked pipeline's input data files are NOT included in this archived
snapshot:

- `runs.jsonl` (METR Time Horizon 1.1 run data, 24,008 runs)
- `benchmark_results_1_1.yaml`

Source: METR `eval-analysis-public`, commit
52cb829c7a2efb2d659285c4b1768d191d97f8d2 (the repository's current HEAD at
the snapshot date), path `reports/time-horizon-1-1/data/raw/runs.jsonl`
for the run data.

Reason for exclusion: at the pinned commit the upstream repository
contains no LICENSE file (its README's License section says only "See
LICENSE file for details"), so no license grant covers redistribution of
the data files in a new venue. They are publicly downloadable from the
source repository; place `runs.jsonl` in this directory to run the full
pipeline (`bash reproduce.sh`). The verifiers (`verify.py`,
`verify_printed.py`) run against this snapshot's deposited outputs.
