from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a structured TopBench CSV output against a target CSV.")
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--id-column", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from topbench.evaluation.structured_output_evaluator import evaluate_id_set_csv

    result = evaluate_id_set_csv(args.prediction_csv, args.target_csv, id_column=args.id_column)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
