import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from transformers.utils import logging
# Suppress HuggingFace hub FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Reduce transformers logging noise
logging.set_verbosity_error()



import pandas as pd
import json
import torch

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from ml_article_tagging.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from ml_article_tagging.data import preprocess, tokenize_data


def main():
    # load datasets
    train_df = pd.read_csv(RAW_DATA_DIR / "train_dataset.csv")
    test_df = pd.read_csv(RAW_DATA_DIR / "test_dataset.csv")

    # create label mapping
    tags = sorted(train_df.tag.unique())
    class_to_index = {tag: i for i, tag in enumerate(tags)}

    # clean text + label encoding
    train_df = preprocess(train_df, class_to_index)
    test_df = preprocess(test_df, class_to_index)

    # split dataset
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["tag"],
        random_state=42
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/scibert_scivocab_uncased"
    )

    # tokenize
    train_data = tokenize_data(train_df, tokenizer)
    val_data = tokenize_data(val_df, tokenizer)
    test_data = tokenize_data(test_df, tokenizer)
    
    # saved processed datasets
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(train_data, PROCESSED_DATA_DIR / "train_data.pt")
    torch.save(val_data, PROCESSED_DATA_DIR / "val_data.pt")
    torch.save(test_data, PROCESSED_DATA_DIR / "test_data.pt")

    # save label mapping
    with open(PROCESSED_DATA_DIR / "class_to_index.json", "w") as f:
        json.dump(class_to_index, f)


    print("\nPreprocessing completed successfully.\n"
)

if __name__ == "__main__":
    main()


