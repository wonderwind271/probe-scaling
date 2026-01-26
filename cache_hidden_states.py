#!/usr/bin/env python3
"""
Extract & store last-token hidden states for Llama-3.1-8B on a HF dataset.

Goal:
- Dataset: Seed42Lab/en-ud-train, split="train", column="text"
- Tokenize + pad/truncate to length 128
- For each example, take the *last non-pad token* (i.e., last token with attention_mask=1)
- Store hidden states from all layers (33 tensors = embeddings + 32 transformer layers)
- Output array shape: (N, 33, 4096)

Storage:
- Uses numpy memmap so you can write incrementally without holding everything in RAM.
- Default dtype float16 to save space (~2.7GB for N=10k).
"""

import os
import json
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Config (edit as you like)
# -----------------------------
DATASET_NAME = "Seed42Lab/en-ud-train"
SPLIT = "train"
TEXT_COL = "text"

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
MAX_LEN = 128

BATCH_SIZE = 8          # adjust based on GPU memory
NUM_WORKERS = 1         # dataloader workers
OUT_DIR = "/scratch/chaijy_root/chaijy2/shuyuwu/llama31_last_token_hiddens"
OUT_DTYPE = np.float32  # np.float16 (smaller) or np.float32 (bigger)

# If you want to resume partially written runs:
RESUME = True


def collate_text(batch: List[Dict]) -> List[str]:
    return [ex[TEXT_COL] for ex in batch]


def load_hidden_states(dir_path: str):
    meta_path = os.path.join(dir_path, "meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    shape = tuple(meta["shape"])
    dtype = np.float16 if "float16" in meta["dtype"] else np.float32
    memmap_path = os.path.join(dir_path, meta["memmap_file"])

    hiddens = np.memmap(
        memmap_path,
        dtype=dtype,
        mode="r",      # read-only (safe)
        shape=shape
    )

    return hiddens, meta



@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load dataset
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    n = len(ds)

    # Optional: if you truly only want ~10k examples
    # ds = ds.select(range(min(10000, len(ds))))
    # n = len(ds)

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # Llama tokenizers often have no pad token by default
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize_texts(texts: List[str]) -> Dict[str, torch.Tensor]:
        return tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    # 3) Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        # If you have multiple GPUs and want HF to shard automatically, uncomment:
        # device_map="auto",
    )
    model.to(device)
    model.eval()

    # Sanity: Llama-3.1-8B hidden size / layers
    hidden_size = model.config.hidden_size         # expect 4096
    num_hidden_layers = model.config.num_hidden_layers  # expect 32
    num_states = num_hidden_layers + 1             # embeddings + each layer => 33

    # 4) Prepare output memmap: (N, 33, 4096)
    out_path = os.path.join(OUT_DIR, "last_token_hidden_states.memmap")
    meta_path = os.path.join(OUT_DIR, "meta.json")
    progress_path = os.path.join(OUT_DIR, "progress.json")

    # Create or open memmap
    if (not RESUME) and os.path.exists(out_path):
        os.remove(out_path)

    mm = np.memmap(
        out_path,
        dtype=OUT_DTYPE,
        mode="r+" if (RESUME and os.path.exists(out_path)) else "w+",
        shape=(n, num_states, hidden_size),
    )

    # Save metadata once
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": DATASET_NAME,
                    "split": SPLIT,
                    "text_col": TEXT_COL,
                    "model": MODEL_NAME,
                    "max_len": MAX_LEN,
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_hidden_layers,
                    "num_states": num_states,
                    "memmap_file": os.path.basename(out_path),
                    "dtype": str(OUT_DTYPE),
                    "shape": [n, num_states, hidden_size],
                },
                f,
                indent=2,
            )

    # Resume bookkeeping
    start_idx = 0
    if RESUME and os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                start_idx = int(json.load(f).get("next_index", 0))
        except Exception:
            start_idx = 0

    # 5) DataLoader
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_text,
        pin_memory=(device == "cuda"),
    )

    # Fast-forward dataloader if resuming
    if start_idx > 0:
        # Skip batches until we reach start_idx
        skip_batches = start_idx // BATCH_SIZE
        # Consume skip_batches batches
        it = iter(loader)
        for _ in range(skip_batches):
            next(it, None)
        loader_iter = it
        batch_offset = skip_batches * BATCH_SIZE
    else:
        loader_iter = iter(loader)
        batch_offset = 0

    # 6) Extract + write
    cur = batch_offset
    pbar_total = (n - cur)

    # simple tqdm without importing (optional). If you want tqdm:
    # from tqdm.auto import tqdm
    # for texts in tqdm(loader_iter, total=(len(loader) - skip_batches if start_idx else len(loader))):
    print(f"[INFO] n={n} | writing to {out_path}")
    print(f"[INFO] hidden_size={hidden_size} | num_states={num_states} | dtype={OUT_DTYPE}")
    print(f"[INFO] starting at index {cur}")

    while True:
        texts = next(loader_iter, None)
        if texts is None:
            break

        bsz = len(texts)
        enc = tokenize_texts(texts)
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(device, non_blocking=True)

        # last non-pad token index per sequence
        # attention_mask sums to actual length; last index = len-1
        last_idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1  # (B,)

        # Forward with hidden states
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states  # tuple length 33, each (B,T,D)

        # Collect last-token for all layers -> (B, 33, D)
        # We gather along T dimension using last_idx
        # For each layer hs: (B,T,D) -> (B,D) at last_idx
        b_last = []
        arange_b = torch.arange(bsz, device=device)
        for hs in hidden_states:
            b_last.append(hs[arange_b, last_idx, :])  # (B,D)
        b_last = torch.stack(b_last, dim=1)  # (B,33,D)

        # Move to CPU + cast for storage
        b_last_np = b_last.to(torch.float16 if OUT_DTYPE == np.float16 else torch.float32).cpu().numpy()

        # Write into memmap
        mm[cur:cur + bsz, :, :] = b_last_np
        mm.flush()

        cur += bsz

        # Persist progress
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"next_index": cur}, f)

        if cur % (BATCH_SIZE * 50) == 0 or cur >= n:
            print(f"[INFO] wrote {cur}/{n}")

        if cur >= n:
            break

    print("[DONE] extraction complete.")
    print(f"[DONE] memmap: {out_path}")
    print(f"[DONE] meta:   {meta_path}")


if __name__ == "__main__":
    main()
