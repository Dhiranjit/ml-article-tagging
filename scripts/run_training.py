import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from transformers.utils import logging
# Suppress HuggingFace hub FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Reduce transformers logging noise
logging.set_verbosity_error()

import argparse
import json
import math
import shutil
from pathlib import Path

import mlflow
import torch
import yaml
from transformers import BertModel, BertTokenizer

from ml_article_tagging.config import PROCESSED_DATA_DIR
from ml_article_tagging.data import create_dataloader
from ml_article_tagging.model import SciBERTClassifier
from ml_article_tagging.train import train
from ml_article_tagging.utils import metric_fn, set_seed, build_scheduler


def main():
    parser = argparse.ArgumentParser(description="Train SciBERT article classifier")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Human-readable experiment identifier (e.g. exp1, lr_1e-4). "
             "Used as the MLflow run name and checkpoint directory name. "
             "If a checkpoint already exists for this run-id, training resumes automatically.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg.get("scheduler", {}) or {}

    # Extract accumulation steps from config, defaulting to 1 safely
    accumulation_steps = train_cfg.get("accumulation_steps", 1)
    
    # Extract the optimization mode (max for f1/accuracy, min for loss)
    metric_mode = train_cfg.get("metric_mode", "max")

    set_seed(exp_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = train_cfg.get("use_amp", False) and device == "cuda"

    model_name = exp_cfg["name"]

    # Load the processed data
    train_data = torch.load(PROCESSED_DATA_DIR / "train_data.pt")
    val_data = torch.load(PROCESSED_DATA_DIR / "val_data.pt")

    with open(PROCESSED_DATA_DIR / "class_to_index.json") as f:
        class_to_index = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(model_cfg["pretrained_name"])

    # Create dataloaders
    train_dataloader = create_dataloader(train_data, tokenizer, batch_size=train_cfg["batch_size"])
    val_dataloader = create_dataloader(val_data, tokenizer, batch_size=train_cfg["batch_size"])

    # Load and initialize 
    scibert = BertModel.from_pretrained(model_cfg["pretrained_name"])

    model = SciBERTClassifier(
        llm=scibert,
        dropout_p=model_cfg["dropout_p"],
        num_classes=len(class_to_index),
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer_cls = getattr(torch.optim, opt_cfg["name"])

    # Filter out the 'name' key, pass the rest directly to the optimizer
    opt_kwargs = {k: v for k, v in opt_cfg.items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

    # Calculate exactly how many times the optimizer will step per epoch
    effective_steps_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)

    scheduler = build_scheduler(
        sched_cfg, optimizer, train_cfg["epochs"], effective_steps_per_epoch
    )

    # Resume logic
    mlflow.set_experiment(model_name)
    experiment = mlflow.get_experiment_by_name(model_name)

    mlflow_run_id = None
    
    if experiment:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{args.run_id}'"
        )
        if not runs.empty: #type: ignore
            mlflow_run_id = runs.iloc[0].run_id # type: ignore
            print(f"Found existing MLflow run '{args.run_id}', resuming...")

    mlflow.set_experiment(model_name)

    with mlflow.start_run(run_id=mlflow_run_id, run_name=args.run_id): #type: ignore
        mlflow.log_artifact(str(args.config), artifact_path="config")

        # --- Run Header ---
        print(f"\n{'='*60}")
        print(f" Run: {args.run_id} | Config: {args.config.name}")
        print(f" bs: {train_cfg['batch_size']} | accum: {accumulation_steps} | "
              f"lr: {opt_cfg['lr']} | epochs: {train_cfg['epochs']} | amp: {use_amp}")
        print(f"{'='*60}\n")

        results = train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=train_cfg["epochs"],
            metric_fn=metric_fn,             
            primary_metric=train_cfg["primary_metric"],
            mode=metric_mode,  # Pass the mode downstream
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
            config=cfg,
            use_amp=use_amp,
        )

        # Load best checkpoint and print validation metrics for the best epoch
        client = mlflow.tracking.MlflowClient() #type: ignore
        local_ckpt_dir = client.download_artifacts(mlflow.active_run().info.run_id, "checkpoints") #type: ignore
        best_ckpt_path = Path(local_ckpt_dir) / "best.pth"
        best_val_metrics = {}
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            best_epoch = best_ckpt["epoch"]
            # Extract validation metrics from the saved results at the best epoch
            saved_results = best_ckpt.get("results", {})
            if saved_results:
                best_val_metrics["val_loss"] = saved_results.get("val_loss", [None])[best_epoch]
                for k, v in saved_results.items():
                    if k.startswith("val_") and k != "val_loss":
                        # pick the value at the best epoch
                        best_val_metrics[k] = v[best_epoch]
            del best_ckpt
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"Loaded best checkpoint (epoch {best_epoch + 1}).")

        # Print validation metrics from the best epoch
        if best_val_metrics:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in best_val_metrics.items() if v is not None)
            print(f"\nBest validation metrics (epoch {best_epoch + 1}): {metrics_str}\n")

        # Save final results as a JSON artifact (no test metrics)
        final_results = {"train": results, "best_val": best_val_metrics}
        
        # Explicit local file handling for Google Colab safety
        tmpdir = Path("local_tmp_results")
        tmpdir.mkdir(exist_ok=True)
        results_path = tmpdir / "results.json"
        
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)
            
        mlflow.log_artifact(str(results_path), artifact_path="results")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()