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

---
Xiaoxi: add different tasks/datasets.
- Somo: use the first 15k in training split.
    python cache_hidden_states.py dataset.name=compling/somo model.max_length=64 dataset.task_name=somo

- factuality:
    python cache_hidden_states.py dataset.name=compling/event_factuality dataset.split=train model.max_length=192 dataset.text_col=sentence dataset.task_name=factuality
"""

import os
import json
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import logging

log = logging.getLogger(__name__)


def collate_text(batch: List[Dict], text_col) -> List[str]:
    return [ex[text_col] for ex in batch]


def get_verb_token_index(batch_encoding, batch_idx, target_word_idx):
    """
    find the last token index corresponding to the target word (predicate) in the original text
    """
    w_ids = batch_encoding.word_ids(batch_index=batch_idx)

    for i in range(len(w_ids) - 1, -1, -1):
        if w_ids[i] == target_word_idx:
            return i
    return None


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


def get_custom_dir(dataset_name, dataset_split, task_name, model_short):
    if dataset_name == 'Seed42Lab/en-ud-test':
        split = 'test'
    elif dataset_name == 'Seed42Lab/en-ud-train':
        split = 'train'
    elif dataset_name == 'compling/somo':
        split = dataset_split
    elif dataset_name == 'compling/event_factuality':
        split = dataset_split
    elif dataset_name == 'compling/event_factuality_balanced':
        split = dataset_split
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')

    return f'cache/{model_short}/{task_name}_{split}'


@hydra.main(version_base=None, config_path=".", config_name="config_cache_hidden")
@torch.no_grad()
def main(cfg: DictConfig):
    out_dir = HydraConfig.get().runtime.output_dir
    logging.info(f"Output directory: {out_dir}")
    save_type = np.float16 if cfg.cache.bf16 else np.float32

    # 1) Load dataset
    ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    if cfg.dataset.name == "compling/somo" and len(ds) > 15000:
        shuffled_ds = ds.shuffle(seed=42)
        ds = shuffled_ds.select(range(15000))
        logging.info(f"Subsetting Somo to 15k examples.")
    if cfg.dataset.task_name == 'factuality':
        columns_to_keep = ["sentence", "label", 'pred_token']
        columns_to_remove = set(ds.column_names) - set(columns_to_keep)
        ds = ds.remove_columns(list(columns_to_remove))
    n = len(ds)

    logging.info(
        f'Loaded dataset: {cfg.dataset.name}, split={cfg.dataset.split}, size={n}')

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
    # Llama tokenizers often have no pad token by default
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize_texts(texts: List[str]) -> Dict[str, torch.Tensor]:
        return tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=cfg.model.max_length,
            return_tensors="pt",
        )

    # 3) Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
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
    out_path = os.path.join(
        out_dir, "last_token_hidden_states.memmap")
    meta_path = os.path.join(out_dir, "meta.json")
    progress_path = os.path.join(out_dir, "progress.json")

    # Create or open memmap
    if (not cfg.cache.resume) and os.path.exists(out_path):
        os.remove(out_path)
        logging.info(f"Removed existing memmap at {out_path} for fresh start.")

    if cfg.dataset.task_name in ["factuality"]:
        hidden_size *= 2

    mm = np.memmap(
        out_path,
        dtype=save_type,
        mode="r+" if (cfg.cache.resume and os.path.exists(out_path)) else "w+",
        shape=(n, num_states, hidden_size),
    )

    # Save metadata once
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": cfg.dataset.name,
                    "split": cfg.dataset.split,
                    "text_col": cfg.dataset.text_col,
                    "model": cfg.model.name,
                    "max_len": cfg.model.max_length,
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_hidden_layers,
                    "num_states": num_states,
                    "memmap_file": os.path.basename(out_path),
                    "dtype": str(save_type),
                    "shape": [n, num_states, hidden_size],
                },
                f,
                indent=2,
            )

    # Resume bookkeeping
    start_idx = 0
    if cfg.cache.resume and os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                start_idx = int(json.load(f).get("next_index", 0))
            logging.info(f"Resuming from index {start_idx}")
        except Exception:
            start_idx = 0

    # 5) DataLoader
    loader = DataLoader(
        ds,
        batch_size=cfg.cache.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=lambda x: x,  # directly return raw dictionary list
        pin_memory=False
    )

    # Fast-forward dataloader if resuming
    if start_idx > 0:
        # Skip batches until we reach start_idx
        skip_batches = start_idx // cfg.cache.batch_size
        # Consume skip_batches batches
        it = iter(loader)
        for _ in range(skip_batches):
            next(it, None)
        loader_iter = it
        batch_offset = skip_batches * cfg.cache.batch_size
    else:
        loader_iter = iter(loader)
        batch_offset = 0
    logging.info(
        f'DataLoader starting at batch offset {batch_offset}.')

    # 6) Extract + write
    cur = batch_offset
    pbar_total = (n - cur)

    # simple tqdm without importing (optional). If you want tqdm:
    # from tqdm.auto import tqdm
    # for texts in tqdm(loader_iter, total=(len(loader) - skip_batches if start_idx else len(loader))):
    logging.info(f"n={n} | writing to {out_path}")
    logging.info(
        f"hidden_size={hidden_size} | num_states={num_states} | dtype={save_type}")
    logging.info(f"starting at index {cur}")

    while True:
        batch_item = next(loader_iter, None)
        if batch_item is None:
            break

        texts = [item[cfg.dataset.text_col] for item in batch_item]
        bsz = len(texts)
        enc = tokenize_texts(texts)
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(device, non_blocking=True)

        last_idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1  # (B,)

        if cfg.dataset.task_name == 'factuality':
            verb_indices = []
            for i, item in enumerate(batch_item):
                assert item['pred_token'] >= 1
                target_word_id = item['pred_token'] - 1
                verb_idx = get_verb_token_index(enc, i, target_word_id)
                if verb_idx is None:
                    raise ValueError('Could not find verb token index for item:', item)
                verb_indices.append(verb_idx)

            verb_indices = torch.tensor(verb_indices, device=device)  # (B,)
            hidden_indices = torch.stack(
                [verb_indices, last_idx], dim=1)  # (B,2)
        else:
            hidden_indices = last_idx.unsqueeze(1)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states

        # For somo and verb-subject agreement tasks, we collect last-token for all layers. For each layer, (B,T,D) -> (B,D) at last_idx. Finally (B,33,D)
        # For facuitality, we concat the final token in the verb's and the last token's hidden state. For each layer, (B,T,D) -> (B,2D). Finally (B,33,2D)
        hidden_embed = []
        for hs in hidden_states:
            dim = hs.size(-1)
            idx_expanded = hidden_indices.unsqueeze(-1).expand(-1, -1, dim)

            # selected[:, idx, :] == hs[torch.arange(128), hidden_indices[:, idx], :]
            selected = torch.gather(hs, 1, idx_expanded).view(
                bsz, -1)  # (B,D) or (B,2D)
            hidden_embed.append(selected)
        hidden_embed = torch.stack(hidden_embed, dim=1)  # (B,33,D)

        # Move to CPU + cast for storage
        hidden_embed_np = hidden_embed.to(
            torch.float16 if cfg.cache.bf16 else torch.float32).cpu().numpy()

        # Write into memmap
        mm[cur:cur + bsz, :, :] = hidden_embed_np
        mm.flush()

        cur += bsz

        # Persist progress
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"next_index": cur}, f)

        if cur % (cfg.cache.batch_size * 50) == 0 or cur >= n:
            print(f"[INFO] wrote {cur}/{n}")

        if cur >= n:
            break

    logging.info("extraction complete.")
    logging.info(f"memmap: {out_path}")
    logging.info(f"meta:   {meta_path}")

    # save labels
    labels = np.asarray(ds[cfg.dataset.label_col], dtype=np.uint8)

    np.save(out_dir + "/labels.npy", labels)

    logging.info(f"[DONE] saved labels to {out_dir}/labels.npy")
    logging.info(f"shape = {labels.shape}, dtype = {labels.dtype}")
    logging.info(f"unique values = {np.unique(labels)}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("calc_path", get_custom_dir)
    main()
