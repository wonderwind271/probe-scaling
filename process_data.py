#!/usr/bin/env python3
"""
Convert two HF datasets of (text,label) arranged as:
  first half: label==1 (positive)
  second half: label==0 (negative)
with 1-1 correspondence: i <-> i+half

into a paired dataset with columns:
  positive, negative

Then upload to: Seed42Lab/en-ud-train-pair
"""

from datasets import load_dataset, Dataset, concatenate_datasets


def to_pairs(repo_id: str, split: str = "train") -> Dataset:
    ds = load_dataset(repo_id, split=split)

    # Basic checks
    assert "text" in ds.column_names and "label" in ds.column_names, ds.column_names
    n = len(ds)
    assert n % 2 == 0, f"{repo_id}: expected even length, got {n}"
    half = n // 2

    labels_first = ds["label"][:half]
    labels_second = ds["label"][half:]
    assert all(int(x) == 1 for x in labels_first), f"{repo_id}: first half not all label==1"
    assert all(int(x) == 0 for x in labels_second), f"{repo_id}: second half not all label==0"

    pos = ds["text"][:half]
    neg = ds["text"][half:]

    paired = Dataset.from_dict({"positive": pos, "negative": neg})
    return paired


def main():
    repos = ["Seed42Lab/en_gum-ud-train", "Seed42Lab/en_ewt-ud-train"]

    paired_list = [to_pairs(r) for r in repos]
    merged = concatenate_datasets(paired_list)

    # Upload
    merged.push_to_hub("Seed42Lab/en-ud-train-pair")  # creates/overwrites the dataset repo


if __name__ == "__main__":
    main()
