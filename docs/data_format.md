# Data Format

TopBench uses four canonical task names:

| Task | Description | Required tables |
| --- | --- | --- |
| `single_point_prediction` | Predict one missing value/category for a described profile. | `history.csv` |
| `decision_making` | Compare multiple candidate scenarios and select the best option. | `history.csv` |
| `treatment_effect_analysis` | Estimate how an intervention changes the outcome. | `history.csv` |
| `ranking_and_filtering` | Produce a structured CSV list from candidate rows. | `history.csv`, `current.csv` |

Query JSON files contain a natural-language `query` and a structured `ground_truth` field used for deterministic scoring after extraction.
