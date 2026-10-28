# Simulator verification

The rewrite retains a local copy of the original simulator at `archived/simulate.py` (not included in Git) and the existing generated dataset.

- Final repository test suite: **26 passed** (including the seed-75 regression case).
- New regression tests compare all four CSVs byte for byte for 800 customers under seeds 42, 7, 123 and 75, using small batches to exercise batch boundaries. They also compare the extracted feature DataFrames, run all 20 data checks and confirm all scam categories occur.
- A separate short-window test covers a run without injected episodes.
- The full 20,000-customer seed-42 run of the current implementation generated four files whose SHA-256 hashes exactly match the existing research dataset. The 216,147-row feature CSV itself was not regenerated in this full-run check; unchanged raw inputs feed the existing extractor. Feature equivalence was checked in the regression tests.
- A 10-customer run through the active command-line entry point and existing feature extractor completed successfully with the current seed-75 configuration. Notebook cell syntax and the active package import also passed.

The full-data file hashes are in `simulator_full_comparison.json`. The original simulator hash is `2b7a744dcadc80db0a4dbc9d04931210cd2f8d819dc77db0da4642f94d2e13cc`.

These tests establish behavioural compatibility for the tested configurations. They do not validate the original simulation assumptions as real-world behaviour.

The full 20,000-customer hash comparison used the original seed-42 configuration and city labels before the researcher changed them. The current YAML uses seed 75 with Pretoria and NMB. The archived and current implementations are compared under the same current configuration in the regression tests. The previous hash comparison does not claim that a seed-75 run matches seed-42 data.
