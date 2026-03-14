import torch
import random
import numpy as np
from transformers import get_scheduler
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score


# ANSI COLORS
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def build_scheduler(sched_cfg: dict, optimizer, epochs: int, steps_per_epoch: int):
    """Returns the instantiated Hugging Face scheduler."""
    sched_name = sched_cfg.get("name")
    if not sched_name:
        return None

    # HF schedulers require total training steps
    num_training_steps = epochs * steps_per_epoch
    
    # Allow passing either absolute warmup steps or a warmup ratio (e.g., 0.1 for 10%)
    warmup_steps = sched_cfg.get("num_warmup_steps", 0)
    warmup_ratio = sched_cfg.get("warmup_ratio", 0.0)
    if warmup_ratio > 0:
        warmup_steps = int(num_training_steps * warmup_ratio)

    # get_scheduler handles "linear", "cosine", "cosine_with_restarts", etc.
    return get_scheduler(
        sched_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def metric_fn(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def set_seed(seed: int) -> None:
    """Sets random seed for reproducibility across multiple libraries.
    
    Args:
        seed (int): The random seed value to set for random, numpy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)

    #PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN settings
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def plot_loss_curves(results: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plots training curves of a model.
    
    Args:
        results (Dict[str, List[float]]): Dictionary containing list of values, e.g.:
            {'train_loss': [...], 'train_acc': [...], 'val_loss': [...], 'val_acc': [...]}
        save_path (Optional[str]): Optional string path to save the figure (e.g., 'plots/results.png').
    """
    
    # Get the loss values of the results dictionary (training and validation)
    loss = results['train_loss']
    test_loss = results['val_loss']

    # Get the accuracy values of the results dictionary (training and validation)
    accuracy = results['train_acc']
    test_accuracy = results['val_acc']

    # Figure out how many epochs there were (start at 1 for the graph)
    epochs = range(1, len(results['train_loss']) + 1)

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Train Loss', marker='o') # Markers help see individual epochs
    plt.plot(epochs, test_loss, label='Val Loss', marker='o')
    plt.title('Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Train Accuracy', marker='o')
    plt.plot(epochs, test_accuracy, label='Val Accuracy', marker='o')
    plt.title('Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        
    plt.show()


### Directory walking utility functions

def build_dir_stats(root: Path) -> Tuple[defaultdict, defaultdict, defaultdict]:
    """Performs a single-pass recursive scan of a directory structure.
    
    Args:
        root (Path): The root directory to scan.
        
    Returns:
        Tuple[defaultdict, defaultdict, defaultdict]: A tuple containing:
            - file_count: Number of files under each path (recursive).
            - dir_size: Total size of files under each path (recursive, in bytes).
            - files_in_dir: Direct file list for each directory (non-recursive).
    """
    file_count = defaultdict(int)
    dir_size = defaultdict(int)
    files_in_dir = defaultdict(list)

    for path in root.rglob("*"):
        if path.is_file():
            size = path.stat().st_size
            parent = path.parent

            # Store direct child files (non-recursive)
            files_in_dir[parent].append(path.name)

            # Add file count + size to all parents
            while True:
                file_count[parent] += 1
                dir_size[parent] += size

                if parent == root:
                    break
                parent = parent.parent

    return file_count, dir_size, files_in_dir


def print_tree(root: Path, file_count: defaultdict, dir_size: defaultdict, 
               files_in_dir: defaultdict, prefix: str = "") -> None:
    """Recursively prints a directory tree with file counts and sizes.
    
    Args:
        root (Path): Current directory to print.
        file_count (defaultdict): Dictionary mapping paths to file counts.
        dir_size (defaultdict): Dictionary mapping paths to total sizes in bytes.
        files_in_dir (defaultdict): Dictionary mapping paths to direct child files.
        prefix (str): String prefix for tree formatting (default: "").
    """
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    dirs = [p for p in entries if p.is_dir()]

    for i, d in enumerate(dirs):
        connector = "├── " if i < len(dirs)-1 else "└── "
        size_mb = dir_size[d] / (1024 * 1024)

        print(
            prefix + connector +
            f"{BLUE}{d.name}/{RESET} "
            f"(files: {GREEN}{file_count[d]}{RESET}, "
            f"size: {YELLOW}{size_mb:.2f} MB{RESET})"
        )

        # If folder has < 15 files → show files (direct children only)
        if file_count[d] < 15 and files_in_dir[d]:
            for file_name in files_in_dir[d]:
                print(prefix + ("│   " if i < len(dirs)-1 else "    ") +
                      f"{CYAN}- {file_name}{RESET}")

        new_prefix = prefix + ("│   " if i < len(dirs)-1 else "    ")
        print_tree(d, file_count, dir_size, files_in_dir, new_prefix)

 
def walk_through_dir(path: str | Path) -> None:
    """Walks through a directory and prints its contents in a tree structure.
    
    Args:
        path (str | Path): The root directory path to walk through.
    """
    root = Path(path)
    print("Scanning directory tree (single pass)...")
    file_count, dir_size, files_in_dir = build_dir_stats(root)

    root_size_mb = dir_size[root] / (1024 * 1024)

    print(
        f"{BLUE}{root}{RESET} "
        f"(files: {GREEN}{file_count[root]}{RESET}, "
        f"size: {YELLOW}{root_size_mb:.2f} MB{RESET})"
    )

    print_tree(root, file_count, dir_size, files_in_dir)
