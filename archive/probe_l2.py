#!/usr/bin/env python3
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


# -------------------------
# Device
# -------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[WARNING] CUDA not available. Running on CPU.")


# -------------------------
# Model + tokenizer
# -------------------------
model_path = "./gpt2_delta_scrub_silu_L2-4/merged_full"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # for batching

model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model.eval()


# -------------------------
# Helpers
# -------------------------
@torch.no_grad()
def last_token_vec(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden: (B, T, D)
    attention_mask: (B, T) with 1 for tokens, 0 for pad
    returns: (B, D) hidden state at last non-pad token
    """
    idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1  # (B,)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]  # (B, D)


def tokenize_pairs(batch, max_length=128):
    pos = tokenizer(
        batch["positive"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    neg = tokenizer(
        batch["negative"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return {
        "pos_input_ids": pos["input_ids"],
        "pos_attention_mask": pos["attention_mask"],
        "neg_input_ids": neg["input_ids"],
        "neg_attention_mask": neg["attention_mask"],
    }


def collate_fn(features):
    # features are already torch-formatted by datasets.set_format
    # each field is shape (B, T)
    batch = {}
    for k in features[0].keys():
        batch[k] = torch.stack([f[k] for f in features], dim=0)
    return batch


@torch.no_grad()
def mean_pairwise_l2_by_layer(model, dataloader, desc=""):
    """
    Computes, for each layer k (including embeddings layer 0):
      mean_i || v_i^{+(k)} - v_i^{-(k)} ||_2
    where v is last-token pooled hidden state.

    Returns: dict layer->mean_l2
    """
    sum_l2 = None
    total = 0

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)

        out_pos = model(
            input_ids=pos_ids,
            attention_mask=pos_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        out_neg = model(
            input_ids=neg_ids,
            attention_mask=neg_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        hs_pos = out_pos.hidden_states  # tuple length = n_layer+1; each (B,T,D)
        hs_neg = out_neg.hidden_states

        if sum_l2 is None:
            sum_l2 = {k: 0.0 for k in range(len(hs_pos))}

        bsz = pos_ids.size(0)
        total += bsz

        for k in range(len(hs_pos)):
            vpos = last_token_vec(hs_pos[k], pos_mask)  # (B,D)
            vneg = last_token_vec(hs_neg[k], neg_mask)  # (B,D)
            # IMPORTANT: L2 per pair first, then mean (we accumulate sum of per-pair L2)
            l2_per_pair = (vpos - vneg).norm(p=2, dim=-1)  # (B,)
            sum_l2[k] += float(l2_per_pair.sum().item())

    mean_l2 = {k: (sum_l2[k] / total) for k in sum_l2}
    return mean_l2


# -------------------------
# Datasets (paired)
# -------------------------
max_len = 128
batch_size = 16

train_pairs = load_dataset("Seed42Lab/en-ud-train-pair", split="train")
test_pairs  = load_dataset("Seed42Lab/en-ud-test-pair",  split="train")

train_pairs = train_pairs.map(
    lambda b: tokenize_pairs(b, max_length=max_len),
    batched=True,
    remove_columns=train_pairs.column_names,
)
train_pairs.set_format(
    type="torch",
    columns=["pos_input_ids", "pos_attention_mask", "neg_input_ids", "neg_attention_mask"],
)

test_pairs = test_pairs.map(
    lambda b: tokenize_pairs(b, max_length=max_len),
    batched=True,
    remove_columns=test_pairs.column_names,
)
test_pairs.set_format(
    type="torch",
    columns=["pos_input_ids", "pos_attention_mask", "neg_input_ids", "neg_attention_mask"],
)

train_loader = DataLoader(train_pairs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_pairs,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# -------------------------
# Compute + print
# -------------------------
train_l2 = mean_pairwise_l2_by_layer(model, train_loader, desc="Train pairs L2")
test_l2  = mean_pairwise_l2_by_layer(model, test_loader,  desc="Test pairs L2")

print("Mean pairwise L2 by layer (train):")
print(train_l2)
print("\nMean pairwise L2 by layer (test):")
print(test_l2)
