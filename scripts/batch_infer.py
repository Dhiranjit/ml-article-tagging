
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()


import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm

from ml_article_tagging.config import PROJECT_ROOT
from ml_article_tagging.predictor import ArticleTagger

# ANSI COLORS for CLI formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def yield_batches(file_path: Path, batch_size: int):
    """
    Reads a JSONL file line by line and yields chunks of size `batch_size`.
    We use a generator here so CPU RAM stays flat, regardless of file size.
    """
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            batch.append(data)
            
            if len(batch) == batch_size:
                yield batch
                batch = []  # Clear the buffer
        
        # Flush the final partial batch if the total count isn't cleanly divisible
        if batch:
            yield batch


def count_lines(file_path: Path) -> int:
    """Utility to count lines quickly for the progress bar."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser(description="Run batch inference on a JSONL file.")
    parser.add_argument(
        "--checkpoint", 
        type=Path, 
        default= PROJECT_ROOT / "best_model/best.pth",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--input", 
        type=Path, 
        required=True,
        help="Path to the input JSONL file containing articles."
    )
    parser.add_argument(
        "--output", 
        type=Path,
        required=True,
        help="Path to save the output JSONL file."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Number of articles to process simultaneously on the GPU."
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"{YELLOW}Error: Input file {args.input} does not exist.{RESET}")
        return

    # 1. Initialize the heavy model once
    tagger = ArticleTagger(args.checkpoint)
    
    total_lines = count_lines(args.input)
    total_batches = (total_lines + args.batch_size - 1) // args.batch_size
    
    print(f"\n{CYAN}Starting batch inference...{RESET}")
    print(f"Total articles: {total_lines}")
    print(f"Batch size: {args.batch_size}")
    print(f"Outputting to: {args.output}\n")

    # Start the execution timer
    start_time = time.time()

    # 2. Open output in 'append' mode. If we crash, we don't lose our work.
    with open(args.output, "a", encoding="utf-8") as out_f:
        
        progress_bar = tqdm(
            yield_batches(args.input, args.batch_size), 
            total=total_batches, 
            desc="Processing Batches"
        )
        
        for batch in progress_bar:
            # 3. Restructure the list of dicts into two lists of strings
            # We use .get() for description so it gracefully falls back to empty strings if missing
            titles = [item["title"] for item in batch]
            descriptions = [item.get("description", "") for item in batch]
            
            # 4. Single forward pass for the whole chunk
            predictions = tagger.predict(titles, descriptions)
            
            # 5. Zip the original data with predictions and flush immediately to disk
            for original_item, pred in zip(batch, predictions):
                original_item["prediction"] = pred["tag"]
                original_item["confidence"] = pred["confidence"]
                
                out_f.write(json.dumps(original_item) + "\n")

    # End the execution timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate throughput safely
    throughput = total_lines / elapsed_time if elapsed_time > 0 else 0

    print(f"\n{GREEN}Batch inference complete!{RESET}")
    print(f"{CYAN}Results saved to:{RESET} {args.output}")
    print(f"{CYAN}Total Items Processed:{RESET} {total_lines}")
    print(f"{CYAN}Total Time Elapsed:{RESET} {elapsed_time:.2f} seconds")
    print(f"{CYAN}Throughput:{RESET} {throughput:.2f} items / second\n")


if __name__ == "__main__":
    main()