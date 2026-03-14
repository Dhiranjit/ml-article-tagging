import pandas as pd
from ml_article_tagging.config import RAW_DATA_DIR

TRAIN_DATA_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
TEST_DATA_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"

def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_data_path = RAW_DATA_DIR / "train_dataset.csv"
    test_data_path = RAW_DATA_DIR / "test_dataset.csv"

    if train_data_path.exists() and test_data_path.exists():
        print("Dataset already exists. Skipping download!!!")
        return
    
    print("Downloading dataset...")

    train_df = pd.read_csv(TRAIN_DATA_URL)
    test_df = pd.read_csv(TEST_DATA_URL)

    train_df.to_csv(train_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)

    print(f"Train Dataset saved to {train_data_path}")
    print(f"Test Dataset saved to {test_data_path}")

if __name__ == "__main__":
    download_data()
