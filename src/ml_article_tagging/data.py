import torch
import numpy as np
from transformers import DataCollatorWithPadding
import re
from torch.utils.data import Dataset, DataLoader


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df, class_to_index):
    df["text"] = df["title"] + " " + df["description"]
    df["text"] = df.text.apply(clean_text) 
    df["tag"] = df["tag"].map(class_to_index)
    
    return df[["text", "tag"]]


def tokenize_data(df, tokenizer):
    """
    Tokenizes the dataset and pre-packs it into a list of dictionaries 
    with PyTorch tensors, ready for the Dataloader.
    """
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=False,
        max_length=512
    )
    labels = df["tag"].tolist()
    # Pack the data upfront during preprocessing
    dataset = []
    for i in range(len(labels)):
        item = {
            key: torch.tensor(val[i]) 
            for key, val in encodings.items()
        }
        item["labels"] = torch.tensor(labels[i])
        dataset.append(item)
        
    return dataset
 

class ArticleDataset(Dataset):
    """
    A mininal dataset wrapper.
    """
    def __init__(self, dataset):
        # dataset is now a simple list of dicts
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # O(1) retrieval
        return self.dataset[idx]
    

def create_dataloader(dataset_list, tokenizer,  batch_size, shuffle=True):
    # dataset_list is the list of dicts loaded from torch.load()
    dataset = ArticleDataset(dataset_list)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )
    
    return dataloader