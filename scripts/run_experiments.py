"""Run a simple hyperparameter sweep over the training script."""

import copy
import itertools
import subprocess
import sys
import tempfile
import os
from pathlib import Path
import yaml


def update_nested_dict(d, key_path, value):
    """Updates a nested dictionary using a dot-separated key path."""
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main():
    base_config_path = Path("configs/scibert.yaml")

    sweep_space = {
        "optimizer.lr": [2e-5, 5e-5],
        "model.dropout_p": [0.1, 0.3, 0.5],
    }

    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    keys = list(sweep_space.keys())
    values_lists = list(sweep_space.values())
    
    total_runs = len(list(itertools.product(*values_lists)))

    for combo_idx, combo_values in enumerate(itertools.product(*values_lists)):
        cfg = copy.deepcopy(base_cfg)
        run_id_parts = []

        for key, val in zip(keys, combo_values):
            update_nested_dict(cfg, key, val)
            
            # Format the run ID string cleanly (e.g., lr-1e-04_dropout_p-0.1)
            param_name = key.split('.')[-1]
            val_str = f"{val:g}" if isinstance(val, float) else str(val)
            run_id_parts.append(f"{param_name}-{val_str}")

        run_id = "_".join(run_id_parts)
        
        # Use a temporary file that automatically cleans itself up
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp)
            run_config_path = tmp.name

        print(f"\n{'='*60}")
        print(f"Starting Sweep Run {combo_idx + 1}/{total_runs}: {run_id}")
        print(f"{'='*60}")

        try:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_training.py",
                    "--config",
                    run_config_path,
                    "--run-id",
                    run_id,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Run {run_id} failed with exit code {e.returncode}. Skipping to next...")
        finally:
            # Explicitly clean up the temp file after MLflow has secured its copy
            try:
                os.remove(run_config_path)
            except OSError:
                pass

    print(f"\n{'='*60}")
    print("All sweep configurations completed. Configurations are saved in MLflow.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()