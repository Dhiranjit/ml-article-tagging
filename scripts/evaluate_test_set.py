import argparse
import json
import warnings
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizer
from transformers.utils import logging

# Suppress HuggingFace hub FutureWarnings & reduce logging noise
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

from ml_article_tagging.config import PROJECT_ROOT, PROCESSED_DATA_DIR
from ml_article_tagging.data import create_dataloader
from ml_article_tagging.model import SciBERTClassifier
from ml_article_tagging.train import eval_model
from ml_article_tagging.utils import metric_fn

# ANSI COLORS for CLI formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a self-contained model artifact on the test set.")
    parser.add_argument(
        "--checkpoint", 
        type=Path, 
        default= PROJECT_ROOT / "best_model/best.pth",
        help="Path to the downloaded best.pth checkpoint"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default= Path("best_model/test_results.json"),
        help="Path to save the final test metrics JSON"
    )
    args = parser.parse_args()

    # 1. Setup & Load Payload
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{CYAN}--- Test Set Evaluation ---{RESET}")
    print(f"Device     : {device.upper()}")
    print(f"Checkpoint : {args.checkpoint}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'. Run select_best_model.py first.")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Extract bundled config
    cfg = checkpoint.get("config")
    if not cfg:
        raise ValueError("The provided checkpoint does not contain a bundled 'config' dictionary.")
        
    model_cfg = cfg["model"]
    batch_size = cfg["training"]["batch_size"]

    # 2. Load Test Data
    test_data_path = PROCESSED_DATA_DIR / "test_data.pt"
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_data_path}.")
    
    print(f"{CYAN}Loading test dataset...{RESET}")
    test_data = torch.load(test_data_path)
    
    with open(PROCESSED_DATA_DIR / "class_to_index.json") as f:
        class_to_index = json.load(f)

    # 3. Create Dataloader
    tokenizer = BertTokenizer.from_pretrained(model_cfg["pretrained_name"])
    test_dataloader = create_dataloader(test_data, tokenizer, batch_size=batch_size)

    # 4. Initialize Model & Load Weights
    print(f"{CYAN}Initializing model architecture and injecting weights...{RESET}")
    scibert = BertModel.from_pretrained(model_cfg["pretrained_name"])
    model = SciBERTClassifier(
        llm=scibert,
        dropout_p=model_cfg["dropout_p"],
        num_classes=len(class_to_index),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 5. Evaluate
    loss_fn = torch.nn.CrossEntropyLoss()
    
    test_results = eval_model(
        model=model, 
        dataloader=test_dataloader, 
        loss_fn=loss_fn, 
        device=device, 
        metric_fn=metric_fn
    )

    # 6. Save Results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(test_results, f, indent=4)
    
    print(f"{GREEN}Test evaluation complete! Metrics saved to: {args.output}{RESET}\n")


if __name__ == "__main__":
    main()