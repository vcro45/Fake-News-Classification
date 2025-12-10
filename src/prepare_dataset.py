# src/prepare_dataset.py

import pandas as pd
import numpy as np
import os

DATA_DIR = "../data"
OUTPUT_FILE = "../data/merged_clean.csv"

def load_and_merge():
    """Load Fake.csv and True.csv, standardize, merge, shuffle."""

    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    print("[INFO] Loading data...")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add label column: 0 = fake, 1 = real
    fake_df["label"] = 0
    true_df["label"] = 1

    # Combine title + text into one feature
    # You can choose text only or title only if you prefer
    print("[INFO] Combining text fields...")
    fake_df["text"] = fake_df["title"].fillna('') + " " + fake_df["text"].fillna('')
    true_df["text"] = true_df["title"].fillna('') + " " + true_df["text"].fillna('')

    # Keep only needed columns
    fake_df = fake_df[["text", "label"]]
    true_df = true_df[["text", "label"]]

    # Merge datasets
    print("[INFO] Merging datasets...")
    merged = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

    # Shuffle the dataset
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[INFO] Dataset merged successfully: {merged.shape[0]} samples")
    return merged


def save_clean_dataset(df):
    """Save merged clean dataset to CSV."""
    print(f"[INFO] Saving cleaned dataset to {OUTPUT_FILE} ...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("[INFO] File saved successfully.")


if __name__ == "__main__":
    df = load_and_merge()
    save_clean_dataset(df)
