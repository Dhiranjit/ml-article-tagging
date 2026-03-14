import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

import argparse
from pathlib import Path
from ml_article_tagging.config import PROJECT_ROOT
from ml_article_tagging.predictor import ArticleTagger

# ANSI COLORS for CLI formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_result(result):
    print(f"\n{YELLOW}--- Prediction Results ---{RESET}")
    display_text = result['cleaned_text'][:150] + "..." if len(result['cleaned_text']) > 150 else result['cleaned_text']
    print(f"{CYAN}Input Text:{RESET} {display_text}")
    print(f"{CYAN}Prediction:{RESET} {GREEN}{result['tag']}{RESET}")
    print(f"{CYAN}Confidence:{RESET} {result['confidence']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a self-contained best.pth model artifact")
    parser.add_argument(
        "--checkpoint", 
        type=Path, 
        default= PROJECT_ROOT / "best_model/best.pth",
        help="Path to the downloaded best.pth checkpoint"
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", type=Path, help="Path to a .txt file containing the article")
    input_group.add_argument("--title", type=str, help="Article title (if not using --file)")
    input_group.add_argument("--interactive", action="store_true", help="Launch a persistent interactive terminal session")
    
    parser.add_argument("--desc", type=str, default="", help="Article description (optional)")
    args = parser.parse_args()


    tagger = ArticleTagger(args.checkpoint)
    
    if args.interactive:
        print(f"\n{GREEN}Interactive mode activated. Model loaded into memory.{RESET}")
        print(f"{YELLOW}Type 'exit' or 'quit' to terminate.{RESET}")
        print(f"{YELLOW}You can paste raw text OR type a path to a .txt file.{RESET}\n")
        
        while True:
            try:
                user_input = input(f"{CYAN}Enter text or file path > {RESET}").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input:
                    continue
                
                # Check if the user typed a valid file path
                input_path = Path(user_input)
                if input_path.is_file():
                    print(f"{CYAN}Reading file: {input_path}{RESET}")
                    with open(input_path, "r", encoding="utf-8") as f:
                        text_to_process = f.read()
                else:
                
                    text_to_process = user_input
                    
                result = tagger.predict(title=text_to_process, description="")
                print_result(result)
                
            except KeyboardInterrupt:
                break
        print(f"\n{YELLOW}Exiting interactive mode.{RESET}")
        return

    # Standard CLI logic
    if args.file:
        if not args.file.exists():
            print(f"{YELLOW}Error: Could not find file at {args.file}{RESET}")
            return
            
        print(f"{CYAN}Reading text from:{RESET} {args.file}")
        with open(args.file, "r", encoding="utf-8") as f:
            raw_content = f.read()
            
        result = tagger.predict(title=raw_content, description="")
    else:
        result = tagger.predict(args.title, args.desc)
    
    print_result(result)


if __name__ == "__main__":
    main()