"""
Training and Evaluation loop for a PyTorch classifier with MLflow integration, 
mixed precision support, gradient accumulation, and checkpointing.
"""

import json
import os
import sys
import shutil
from pathlib import Path
import torch
from tqdm.auto import tqdm
import mlflow
from mlflow.tracking import MlflowClient


# ANSI COLORS
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def load_checkpoint(filename, model, optimizer, device, scheduler=None, scaler=None):
    """Loads state and returns metadata (epoch, results, best_score)."""
    print(f"\n{CYAN}Loading checkpoint: {filename}{RESET}\n")
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    return checkpoint


def train_step(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    epoch_index,
    total_epochs,
    scheduler=None,
    accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None, #type: ignore
):
    model.train()
    train_loss = 0
    use_amp = scaler is not None and scaler.is_enabled()

    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch_index + 1}/{total_epochs}]")

    # 1. Start with clean gradients before the loop
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(progress_bar, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["labels"]

        # AMP: autocast the forward pass and loss computation
        with torch.amp.autocast('cuda', enabled=use_amp): #type: ignore
            y_pred = model(batch)
            loss = loss_fn(y_pred, y)

        # 2. Scale the loss down by the accumulation factor
        scaled_loss = loss / accumulation_steps
        
        # 3. Accumulate gradients (scaler is a no-op when AMP is disabled)
        if scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # 4. Step only when we hit the accumulation threshold, or at the very end of the epoch
        if batch_idx % accumulation_steps == 0 or batch_idx == len(dataloader):
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # type: ignore
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            if scheduler:
                scheduler.step()
                
            # Clear gradients ONLY after we step
            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{train_loss / batch_idx:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return train_loss / len(dataloader)


def val_step(model, dataloader, loss_fn, device, metric_fn=None):
    model.eval()
    val_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch["labels"]

            y_pred = model(batch)
            preds = y_pred.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            val_loss += loss_fn(y_pred, y).item()

    metrics = metric_fn(all_labels, all_preds) if metric_fn is not None else {}

    return val_loss / len(dataloader), metrics


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    device,
    epochs,
    metric_fn,
    primary_metric,
    scheduler=None,
    config: dict | None = None,
    accumulation_steps: int = 1,
    use_amp: bool = False,
    mode: str = "max",
):
    # Initialize State
    start_epoch = 0
    # Setup correct infinity bound based on mode
    best_val_metric = float("inf") if mode == "min" else float("-inf")
    results = {"train_loss": [], "val_loss": []}
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) #type:ignore

    active_run = mlflow.active_run()
    client = MlflowClient()

    # Resume Logic via MLflow
    try:
        local_checkpoint_dir = client.download_artifacts(active_run.info.run_id, "checkpoints") # type: ignore
        path_last = Path(local_checkpoint_dir) / "last.pth"
        
        if path_last.exists():
            checkpoint = load_checkpoint(path_last, model, optimizer, device, scheduler, scaler=scaler)
            start_epoch = checkpoint['epoch'] + 1
            best_val_metric = checkpoint.get('best_val_metric', best_val_metric)

            if 'results' in checkpoint:
                results = checkpoint['results']

            del checkpoint
            torch.cuda.empty_cache()

            print(f"{GREEN}Resuming from Epoch {start_epoch}{RESET}")

            if start_epoch >= epochs:
                print(f"{YELLOW}Warning: Targeted epochs ({epochs}) is less than "
                      f"already completed epochs ({start_epoch}).{RESET}")
                print(f"{YELLOW}Nothing to do. Exiting...{RESET}")
                return results
    except Exception:
        print(f"{YELLOW}Starting fresh training run. No checkpoint found in MLflow.{RESET}")

    # Log Params
    if start_epoch == 0:
        runtime_params = {
            "epochs": epochs,
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__ if scheduler else "None",
            "loss_fn": type(loss_fn).__name__,
            "primary_metric": primary_metric,
            "metric_mode": mode,
            "initial_lr": optimizer.param_groups[0]["lr"],
        }
        mlflow.log_params(runtime_params)

    try:
        for epoch in range(start_epoch, epochs):
            # --- Train Step ---
            train_loss = train_step(
                model, 
                train_dataloader, 
                loss_fn, 
                optimizer, 
                device, 
                epoch, 
                epochs,
                scheduler=scheduler,
                accumulation_steps=accumulation_steps,
                scaler=scaler)
            
            # --- Val Step ---
            val_loss, metrics = val_step(
                model, val_dataloader, loss_fn, device, metric_fn=metric_fn
            )

            # --- Update Metrics ---
            results["train_loss"].append(train_loss)
            results["val_loss"].append(val_loss)
            for k, v in metrics.items():
                results.setdefault(f"val_{k}", []).append(v)

            # --- Printing ---
            padding = " " * (len(f"Epoch [{epoch + 1}/{epochs}]") + 2)
            metrics_str = " | ".join(
                f"{CYAN}val_{k}:{RESET} {GREEN}{v:.4f}{RESET}" for k, v in metrics.items()
            )
            print(
                f"{padding}{CYAN}train_loss:{RESET} {YELLOW}{train_loss:.4f}{RESET} | "
                f"{CYAN}val_loss:{RESET} {YELLOW}{val_loss:.4f}{RESET}"
                + (f" | {metrics_str}" if metrics_str else "")
            )

            # --- Log Metrics to MLflow ---
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss,
                 **{f"val_{k}": v for k, v in metrics.items()}},
                step=epoch,
            )

            # --- Check if Best ---
            # Default to infinity bounds if metric is somehow missing from this epoch
            fallback_val = float("inf") if mode == "min" else float("-inf")
            current_metric = metrics.get(primary_metric, fallback_val)
            
            is_best = (current_metric < best_val_metric) if mode == "min" else (current_metric > best_val_metric)
            
            if is_best:
                best_val_metric = current_metric
                
                # Log the best metrics to MLflow
                best_metrics_to_log = {"best_val_loss": val_loss}
                best_metrics_to_log.update({f"best_val_{k}": v for k, v in metrics.items()})
                mlflow.log_metrics(best_metrics_to_log, step=epoch)

            # --- Construct State Dict ---
            active_run = mlflow.active_run()
            current_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'best_val_metric': best_val_metric,
                'results': results,
                'mlflow_run_id': active_run.info.run_id if active_run else None,
                'config': config,
            }

            # --- Save Checkpoints directly to MLflow ---
            # Avoid `tempfile` context manager which can aggressively auto-delete in Colab
            # before MLflow network uploads finish. We explicitly manage a local dir.
            tmpdir = Path("local_tmp_checkpoints")
            tmpdir.mkdir(parents=True, exist_ok=True)
            
            path_last = tmpdir / "last.pth"
            torch.save(current_state, path_last)
            mlflow.log_artifact(str(path_last), artifact_path="checkpoints")

            if is_best:
                print(f"{GREEN}>>> Best model updated{RESET}")
                path_best = tmpdir / "best.pth"
                torch.save(current_state, path_best)
                mlflow.log_artifact(str(path_best), artifact_path="checkpoints")
            
            # Explicit cleanup after MLflow upload succeeds
            shutil.rmtree(tmpdir, ignore_errors=True)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Training interrupted by user. Last checkpoint is safely in MLflow.{RESET}")
        sys.exit(0)

    print(f"{GREEN}Training complete! Artifacts saved to MLflow.{RESET}")
    return results


def eval_model(model, dataloader, loss_fn, device, metric_fn):
    print(f"{CYAN}Evaluating model on test set...{RESET}")
    
    # Reuse val_step logic to ensure consistent metric calculation
    test_loss, metrics = val_step(model, dataloader, loss_fn, device, metric_fn=metric_fn)

    metrics_str = " | ".join(
        f"{CYAN}Test {k}:{RESET} {GREEN}{v:.4f}{RESET}" for k, v in metrics.items()
    )
    
    print(
        f"\n{CYAN}Test Loss:{RESET} {YELLOW}{test_loss:.4f}{RESET} | "
        f"{metrics_str}\n"
    )

    return {"loss": test_loss, **metrics}