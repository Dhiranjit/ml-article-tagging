
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()



import os
from pathlib import Path
from typing import List, Optional

import ray
import torch
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel

from ml_article_tagging.config import PROJECT_ROOT
from ml_article_tagging.predictor import ArticleTagger

# ANSI COLORS for local terminal logging
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# 1. Define the API Schema using Pydantic
# This guarantees that the HTTP endpoint rejects malformed JSON before it ever hits our model.
app = FastAPI(
    title="SciBERT Article Tagger API",
    description="Online inference endpoint for ML article classification."
)

class ArticleRequest(BaseModel):
    title: str
    description: Optional[str] = ""

class ArticleResponse(BaseModel):
    tag: str
    confidence: float
    cleaned_text: str

NUM_GPUS = 1 if torch.cuda.is_available() else 0

# 2. Define the Ray Serve Deployment Worker
@serve.deployment(
    num_replicas=1, 
    ray_actor_options={"num_gpus": NUM_GPUS}
)
@serve.ingress(app)
class ArticleTaggerDeployment:
    def __init__(self, checkpoint_path: str):
        """
        Initialization runs exactly ONCE when the Ray worker boots up.
        This is where we load the heavy transformer weights into VRAM.
        """
        print(f"{CYAN}Booting Ray Serve Worker and loading model...{RESET}")
        checkpoint = Path(checkpoint_path)
        self.tagger = ArticleTagger(checkpoint)
        print(f"{GREEN}Worker ready.{RESET}")

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.05)
    async def process_batch(self, requests: List[ArticleRequest]) -> List[ArticleResponse]:
        """
        The dynamic batching core. Ray Serve accumulates incoming requests here.
        Instead of processing 1 request at a time, we process up to 16 at once.
        """
        # Extract the fields from the Pydantic models
        titles = [req.title for req in requests]
        descriptions = [req.description for req in requests]
        
        # Execute the single forward pass on the GPU using our batched predictor
        results = self.tagger.predict(titles, descriptions)
        
        # Repackage the raw dictionaries back into our Pydantic response models
        return [ArticleResponse(**res) for res in results]

    @app.post("/predict", response_model=ArticleResponse)
    async def predict(self, request: ArticleRequest):
        """
        The actual HTTP endpoint. 
        Notice it accepts a SINGLE request, but awaits the BATCHED processor.
        """
        # Ray Serve suspends execution here, tosses the request into the batch queue,
        # waits for the window to close, runs the GPU pass, and returns the specific result.
        return await self.process_batch(request)


# 3. Bind the deployment
# This is the entrypoint that Ray Serve looks for when launching the application.
_default_checkpoint = str(PROJECT_ROOT / "best_model/best.pth")
tagger_app = ArticleTaggerDeployment.bind(checkpoint_path=_default_checkpoint)