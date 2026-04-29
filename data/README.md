# TopBench Data

The full TopBench dataset is not stored in this GitHub repository.

After the dataset is released on HuggingFace, place or symlink it here with this layout:

```text
data/
  single_point_prediction/
  decision_making/
  treatment_effect_analysis/
  ranking_and_filtering/
```

Each dataset folder should contain:

- `history.csv`: historical training table.
- `current.csv`: candidate table, required by Ranking and Filtering.
- `info.json` or `info_mod.json`: target and schema metadata.
- `*.json`: natural-language query files with structured ground truth.

Legacy names are supported for local reproduction only:

- `B1` -> `single_point_prediction`
- `B2` -> `decision_making`
- `B3` -> `treatment_effect_analysis`
- `B4` -> `ranking_and_filtering`
