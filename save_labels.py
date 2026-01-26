#!/usr/bin/env python3
"""
Extract labels from Seed42Lab/en-ud-train and save as labels.npy

Assumes:
- hidden states were saved in the same dataset order
- column "label" exists and is 0/1
"""

import numpy as np
from datasets import load_dataset

DATASET_NAME = "Seed42Lab/en-ud-test"
SPLIT = "train"
LABEL_COL = "label"

OUT_PATH = "llama31_last_token_hiddens_eval/labels.npy"


def main():
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    # Direct, zero-fuss extraction
    labels = np.asarray(ds[LABEL_COL], dtype=np.uint8)

    np.save(OUT_PATH, labels)

    print(f"[DONE] saved labels to {OUT_PATH}")
    print(f"[INFO] shape = {labels.shape}, dtype = {labels.dtype}")
    print(f"[INFO] unique values = {np.unique(labels)}")


if __name__ == "__main__":
    main()
