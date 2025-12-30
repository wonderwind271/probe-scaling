#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, default_data_collator
from transformers.models.gpt2.modeling_gpt2 import Conv1D


# -------------------------
# Delta wrappers (switchable)
# -------------------------

class DeltaConv1D(nn.Module):
    def __init__(self, base: Conv1D, train_bias: bool = True, init_scale: float = 0.0):
        super().__init__()
        self.nf = base.nf
        self.register_buffer("W0", base.weight.detach().clone())
        self.register_buffer("b0", base.bias.detach().clone())

        self.dW = nn.Parameter(torch.zeros_like(self.W0))
        if init_scale != 0.0:
            nn.init.normal_(self.dW, mean=0.0, std=init_scale)

        if train_bias:
            self.db = nn.Parameter(torch.zeros_like(self.b0))
        else:
            self.db = nn.Parameter(torch.zeros_like(self.b0), requires_grad=False)

        self.register_buffer("alpha", torch.tensor(1.0))  # 1 on, 0 off

    def set_delta_enabled(self, enabled: bool):
        self.alpha.fill_(1.0 if enabled else 0.0)

    def set_delta_trainable(self, trainable: bool):
        self.dW.requires_grad_(trainable)
        if isinstance(self.db, nn.Parameter):
            self.db.requires_grad_(trainable)

    def forward(self, x):
        x_2d = x.view(-1, x.size(-1))
        W = self.W0 + self.alpha * self.dW
        b = self.b0 + self.alpha * self.db
        y = torch.addmm(b, x_2d, W)
        return y.view(*x.size()[:-1], self.nf)


class DeltaLayerNorm(nn.Module):
    def __init__(self, base: nn.LayerNorm, init_scale: float = 0.0):
        super().__init__()
        self.normalized_shape = base.normalized_shape
        self.eps = base.eps
        self.elementwise_affine = base.elementwise_affine

        if self.elementwise_affine:
            self.register_buffer("g0", base.weight.detach().clone())
            self.register_buffer("b0", base.bias.detach().clone())
            self.dg = nn.Parameter(torch.zeros_like(self.g0))
            self.db = nn.Parameter(torch.zeros_like(self.b0))
            if init_scale != 0.0:
                nn.init.normal_(self.dg, mean=0.0, std=init_scale)
                nn.init.normal_(self.db, mean=0.0, std=init_scale)
        else:
            self.register_buffer("g0", None)
            self.register_buffer("b0", None)
            self.dg = None
            self.db = None

        self.register_buffer("alpha", torch.tensor(1.0))

    def set_delta_enabled(self, enabled: bool):
        self.alpha.fill_(1.0 if enabled else 0.0)

    def set_delta_trainable(self, trainable: bool):
        if self.dg is not None:
            self.dg.requires_grad_(trainable)
        if self.db is not None:
            self.db.requires_grad_(trainable)

    def forward(self, x):
        if not self.elementwise_affine:
            return F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        g = self.g0 + self.alpha * self.dg
        b = self.b0 + self.alpha * self.db
        return F.layer_norm(x, self.normalized_shape, g, b, self.eps)


# -------------------------
# Replace GPT-2 block modules with Delta versions
# -------------------------

def freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def replace_modules_with_delta(module: nn.Module, init_scale: float = 0.0):
    for name, child in list(module.named_children()):
        replace_modules_with_delta(child, init_scale=init_scale)

        if isinstance(child, Conv1D):
            setattr(module, name, DeltaConv1D(child, train_bias=True, init_scale=init_scale))
        elif isinstance(child, nn.LayerNorm):
            setattr(module, name, DeltaLayerNorm(child, init_scale=init_scale))
        # GPT-2 mainly uses Conv1D + LayerNorm; Linear is rare unless you added your own.


def enable_delta_for_layers(model: GPT2LMHeadModel, layers: List[int], init_scale: float = 0.0):
    """
    Freeze everything, then convert specified transformer blocks h[layer] into delta blocks.
    """
    freeze_all_params(model)
    for L in layers:
        replace_modules_with_delta(model.transformer.h[L], init_scale=init_scale)

    # Enable training for delta params inside those layers
    for L in layers:
        for m in model.transformer.h[L].modules():
            if hasattr(m, "set_delta_trainable"):
                m.set_delta_trainable(True)
    return model


def set_layers_delta_enabled(model: GPT2LMHeadModel, layers: List[int], enabled: bool):
    for L in layers:
        for m in model.transformer.h[L].modules():
            if hasattr(m, "set_delta_enabled"):
                m.set_delta_enabled(enabled)


def delta_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


# -------------------------
# Merge back to normal GPT-2 modules for save_pretrained()
# -------------------------

def _merge_delta_conv1d(m: DeltaConv1D):
    alpha = float(m.alpha.item())
    W = (m.W0 + alpha * m.dW).detach()
    b = (m.b0 + alpha * m.db).detach()
    out = Conv1D(m.nf, W.shape[0])
    out.weight.data.copy_(W)
    out.bias.data.copy_(b)
    return out

def _merge_delta_layernorm(m: DeltaLayerNorm):
    out = nn.LayerNorm(m.normalized_shape, eps=m.eps, elementwise_affine=m.elementwise_affine)
    if not m.elementwise_affine:
        return out
    alpha = float(m.alpha.item())
    g = (m.g0 + alpha * m.dg).detach()
    b = (m.b0 + alpha * m.db).detach()
    out.weight.data.copy_(g)
    out.bias.data.copy_(b)
    return out

def merge_delta_modules_inplace(module: nn.Module):
    for name, child in list(module.named_children()):
        merge_delta_modules_inplace(child)

        cls = child.__class__.__name__
        if cls == "DeltaConv1D":
            setattr(module, name, _merge_delta_conv1d(child))
        elif cls == "DeltaLayerNorm":
            setattr(module, name, _merge_delta_layernorm(child))
    return module


# -------------------------
# Data
# -------------------------

@dataclass
class PairBatch:
    pos_input_ids: torch.Tensor
    pos_attention_mask: torch.Tensor
    neg_input_ids: torch.Tensor
    neg_attention_mask: torch.Tensor

def tokenize_pairs(ds, tokenizer: GPT2TokenizerFast, max_len: int = 128):
    # GPT-2 has no pad by default; set to eos for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tok(examples):
        pos = tokenizer(
            examples["positive"], truncation=True, padding="max_length", max_length=max_len
        )
        neg = tokenizer(
            examples["negative"], truncation=True, padding="max_length", max_length=max_len
        )
        return {
            "pos_input_ids": pos["input_ids"],
            "pos_attention_mask": pos["attention_mask"],
            "neg_input_ids": neg["input_ids"],
            "neg_attention_mask": neg["attention_mask"],
        }

    out = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    out.set_format(type="torch")
    return out


def collate_fn(features: List[Dict[str, Any]]) -> PairBatch:
    batch = default_data_collator(features)
    return PairBatch(
        pos_input_ids=batch["pos_input_ids"],
        pos_attention_mask=batch["pos_attention_mask"],
        neg_input_ids=batch["neg_input_ids"],
        neg_attention_mask=batch["neg_attention_mask"],
    )


# -------------------------
# Loss pieces
# -------------------------

def last_token_index(attn_mask: torch.Tensor) -> torch.Tensor:
    # attn_mask: (B, T) with 1 for tokens, 0 for pad
    # returns (B,) index of last non-pad token
    return attn_mask.long().sum(dim=1).clamp(min=1) - 1

def select_last_hidden(hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # hidden: (B,T,D)
    idx = last_token_index(attn_mask)  # (B,)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]  # (B,D)

def kl_to_base_logits(logits: torch.Tensor, base_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    KL( p_theta || p_base ) averaged over tokens where mask==1.
    logits/base_logits: (B,T,V)
    mask: (B,T)
    """
    logp = F.log_softmax(logits, dim=-1)
    logq = F.log_softmax(base_logits, dim=-1)
    p = logp.exp()
    kl = (p * (logp - logq)).sum(dim=-1)  # (B,T)
    denom = mask.sum().clamp(min=1)
    return (kl * mask).sum() / denom


# -------------------------
# Training
# -------------------------

def train(
    repo_id: str = "Seed42Lab/en-ud-train-pair",
    base_model_name: str = "gpt2",
    scrub_layers: List[int] = (2, 3, 4),
    max_len: int = 128,
    batch_size: int = 16,
    lr: float = 1e-4,
    steps: int = 5000,
    grad_accum: int = 1,
    lambda_conf: float = 1.0,
    lambda_kl: float = 0.1,
    lambda_delta: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir: str = "./gpt2_delta_scrub_L2-4",
):
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(repo_id, split="train")
    ds = tokenize_pairs(ds, tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    base = GPT2LMHeadModel.from_pretrained(base_model_name).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    model = GPT2LMHeadModel.from_pretrained(base_model_name).to(device)
    model = enable_delta_for_layers(model, list(scrub_layers), init_scale=0.0)
    set_layers_delta_enabled(model, list(scrub_layers), enabled=True)
    model.train()

    # Optimizer over trainable params only (deltas)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Simple layer weights: ramp up toward max scrub layer
    Lmax = max(scrub_layers)
    # w_k = exp(beta*(k-Lmax)); beta=0.5
    beta = 0.5
    layer_w = {k: math.exp(beta * (k - Lmax)) for k in scrub_layers}

    def delta_l2():
        s = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                s = s + (p.pow(2).sum())
        return s

    step = 0
    opt.zero_grad(set_to_none=True)

    while step < steps:
        for batch in dl:
            if step >= steps:
                break

            pos_ids = batch.pos_input_ids.to(device)
            pos_mask = batch.pos_attention_mask.to(device)
            neg_ids = batch.neg_input_ids.to(device)
            neg_mask = batch.neg_attention_mask.to(device)

            # Forward edited model (need hidden states + logits)
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

            # Base logits for KL anchoring (no grad)
            with torch.no_grad():
                base_pos = base(input_ids=pos_ids, attention_mask=pos_mask, use_cache=False).logits
                base_neg = base(input_ids=neg_ids, attention_mask=neg_mask, use_cache=False).logits

            # Confusion loss on selected layers, last non-pad token
            conf = 0.0
            for k in scrub_layers:
                hpos_k = out_pos.hidden_states[k]  # (B,T,D) note: hidden_states[0] is embeddings
                hneg_k = out_neg.hidden_states[k]
                vpos = select_last_hidden(hpos_k, pos_mask)  # (B,D)
                vneg = select_last_hidden(hneg_k, neg_mask)
                conf = conf + layer_w[k] * F.mse_loss(vpos, vneg)

            # KL anchoring (all tokens)
            kl = 0.5 * (
                kl_to_base_logits(out_pos.logits, base_pos, pos_mask) +
                kl_to_base_logits(out_neg.logits, base_neg, neg_mask)
            )

            # Delta L2
            reg = delta_l2()

            loss = lambda_conf * conf + lambda_kl * kl + lambda_delta * reg
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % 200 == 0:
                print(
                    f"step {step:5d} | loss {loss.item()*grad_accum:.4f} "
                    f"| conf {conf.item():.4f} | kl {kl.item():.4f} | reg {reg.item():.2e}"
                )

            step += 1

    # Save delta-only
    torch.save(delta_state_dict(model), os.path.join(out_dir, "delta_only.pt"))

    # Merge and save full model in normal HF format
    merged = model.to("cpu")
    merge_delta_modules_inplace(merged)
    merged.save_pretrained(os.path.join(out_dir, "merged_full"))
    tokenizer.save_pretrained(os.path.join(out_dir, "merged_full"))

    print(f"Saved delta_only.pt and merged_full/ to {out_dir}")


if __name__ == "__main__":
    train()
