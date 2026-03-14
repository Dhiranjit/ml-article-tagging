# SciBERT Text Classification Pipeline

A complete pipeline for **training, evaluating, and deploying SciBERT-based model** for ml article tagging.  
The project includes data preparation, experiment management, model selection, inference, and online serving.

---

## Project Overview

This repository provides an end-to-end workflow for building a SciBERT classifier:

- Data download and preprocessing
- Training with configurable experiments
- Automated experiment comparison
- Best model selection
- Evaluation on test data
- CLI inference and batch inference
- Online model serving

---

## Setup

Install dependencies and prepare the dataset.

```bash
pip install -r requirements.txt

python scripts/get_data.py
python scripts/preprocess.py
```

---

# Training

### Run a Single Experiment

```bash
python scripts/run_training.py \
  --config configs/scibert.yaml \
  --run-id scibert_lr_2e-5_dp_0.3
```

### Run Multiple Experiments

```bash
python scripts/run_experiments.py
```

---

# Model Selection

After running experiments, select the best model automatically.

```bash
python scripts/select_best_model.py --experiment scibert
```

The best model checkpoint will be stored in:

```
/best_model/
```

---

# Evaluation

### Evaluate Best Model

```bash
python scripts/evaluate_test_set.py
```

### Evaluate a Specific Checkpoint

```bash
python scripts/evaluate_test_set.py \
  --checkpoint best_model/best.pth
```

---

# Inference

### Interactive CLI

```bash
python scripts/infer.py --interactive
```

### Direct Input

```bash
python scripts/infer.py \
  --title "Attention Is All You Need" \
  --desc "We propose..."
```

### Batch Inference (JSONL)

```bash
python scripts/batch_infer.py \
  --input data/raw/test_data.jsonl \
  --output outputs/test_predictions.jsonl \
  --batch-size 32
```

---

# Model Serving

Start the online inference API.

```bash
serve run scripts.serve_online:tagger_app
```

Test the serving endpoint:

```bash
python tests/test_serve.py
```

---

# Project Structure

```
├── configs/
│ └── scibert.yaml

├── data/
│ ├── raw/ # original datasets
│ └── processed/ # tokenized / tensor datasets

├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing_and_tokenization.ipynb
│ └── 03_model.ipynb

├── scripts/
│ ├── get_data.py
│ ├── preprocess.py
│ ├── run_training.py
│ ├── run_experiments.py
│ ├── select_best_model.py
│ ├── evaluate_test_set.py
│ ├── infer.py
│ ├── batch_infer.py
│ └── serve_online.py

├── src/
│ └── ml_article_tagging/
│ ├── config.py
│ ├── data.py
│ ├── model.py
│ ├── predictor.py
│ ├── train.py
│ └── utils.py

├── tests/
│ └── test_serve.py

├── outputs/ # predictions and experiment outputs
├── texts/ # example input articles

└── README.md
```
