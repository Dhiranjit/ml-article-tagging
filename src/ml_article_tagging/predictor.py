import json
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizer

from ml_article_tagging.config import PROCESSED_DATA_DIR
from ml_article_tagging.data import clean_text 
from ml_article_tagging.model import SciBERTClassifier

# ANSI COLORS for CLI formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


class ArticleTagger:
    """
    A persistent inference engine. Loads the heavy model weights once into VRAM 
    and exposes a fast .predict() method for individual or batched articles.
    """
    def __init__(self, checkpoint_path: Path, device: str | None = None):
        # 1. Setup Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{CYAN}Initializing inference engine on: {self.device.upper()}{RESET}")
        
        # 2. Load the Local Checkpoint FIRST
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Did you run select_best_model.py?")
            
        print(f"{CYAN}Loading artifact payload from: {checkpoint_path}{RESET}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 3. Extract the bundled configuration
        self.cfg = checkpoint.get("config")
        if not self.cfg:
            raise ValueError("The provided checkpoint does not contain a bundled 'config' dictionary.")
            
        model_cfg = self.cfg["model"]
        
        # 4. Load Label Mapping (Reverse it for inference: Index -> Class)
        with open(PROCESSED_DATA_DIR / "class_to_index.json") as f:
            class_to_index = json.load(f)
        self.index_to_class = {v: k for k, v in class_to_index.items()}
        
        # 5. Load Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_cfg["pretrained_name"])
        
        # 6. Initialize Model Architecture using the bundled config
        scibert = BertModel.from_pretrained(model_cfg["pretrained_name"])
        self.model = SciBERTClassifier(
            llm=scibert,
            dropout_p=model_cfg["dropout_p"],
            num_classes=len(self.index_to_class)
        ).to(self.device)
        
        # 7. Inject the weights into the architecture
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # 8. Lock the model for inference (disables Dropout and locks BatchNorm)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, title: str | list[str], description: str | list[str] = ""):
        """Runs articles through the pipeline. Supports single strings or lists of strings."""
        is_single = isinstance(title, str)
        
        # Normalize inputs to lists
        titles = [title] if is_single else title
        
        if is_single:
            descriptions = [description]
        elif isinstance(description, str):
            descriptions = [description] * len(titles)
        else:
            descriptions = description
            
        raw_texts = [f"{t} {d}".strip() for t, d in zip(titles, descriptions)]
        cleaned_texts = [clean_text(text) for text in raw_texts]
        
        # Tokenizer needs padding enabled for batches
        inputs = self.tokenizer(
            cleaned_texts,
            padding=True, 
            truncation=True,
            max_length=512,
            return_tensors="pt" 
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        logits = self.model(inputs)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidences, predicted_indices = torch.max(probs, dim=1)
        
        results = []
        for i in range(len(predicted_indices)):
            tag = self.index_to_class[predicted_indices[i].item()]
            results.append({
                "tag": tag,
                "confidence": confidences[i].item(),
                "cleaned_text": cleaned_texts[i]
            })
        
        # Return single dict if input was single, else list of dicts
        return results[0] if is_single else results