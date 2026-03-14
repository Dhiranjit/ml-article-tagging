import argparse
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best run by metric and download its 'best.pth' checkpoint."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="MLflow experiment name (e.g. scibert)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="best_val_f1",
        help="Metric name to rank by (default: best_val_f1)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["max", "min"],
        default="max",
        help="Whether a higher or lower metric is better (default: max)",
    )
    args = parser.parse_args()

    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise ValueError(f"Experiment '{args.experiment}' not found.")

    # 1. Fetch all runs for the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        raise ValueError("No runs found in this experiment.")

    metric_col = f"metrics.{args.metric}"
    if metric_col not in runs.columns:
        available = [c.replace("metrics.", "") for c in runs.columns if c.startswith("metrics.")]
        raise ValueError(f"Metric '{args.metric}' not found. Available metrics: {available}")

    # 2. Filter out missing data and sort
    ranked = runs.dropna(subset=[metric_col]).sort_values(
        by=metric_col, 
        ascending=(args.mode == "min")
    )

    if ranked.empty:
        raise ValueError(f"No valid data found for metric '{args.metric}'.")

    # 3. Extract the winner
    best_run = ranked.iloc[0]
    best_run_id = best_run["run_id"]
    best_metric_value = best_run[metric_col]
    run_name = best_run.get("tags.mlflow.runName", "Unnamed")

    print(f"Best run_name : {run_name}")
    print(f"Best run_id   : {best_run_id}")
    print(f"{args.metric:<13}: {best_metric_value:.4f}")

    # 4. Download checkpoint to local directory
    client = MlflowClient()
    output_dir = Path("best_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoint to {output_dir.resolve()}...")
    try:
        local_path = client.download_artifacts(
            run_id=best_run_id,
            path="checkpoints/best.pth",
            dst_path=str(output_dir),
        )
        print(f"Success! Model checkpoint saved to: {local_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to download artifact. Make sure 'checkpoints/best.pth' exists in run {best_run_id}. Error: {exc}")


if __name__ == "__main__":
    main()