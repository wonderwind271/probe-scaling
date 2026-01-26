from __future__ import annotations

from probe_model import MLP, MultiLinear
import os
import json
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)


# -----------------------------
# Cache loading
# -----------------------------
@dataclass
class CacheMeta:
    memmap_file: str
    shape: Tuple[int, int, int]   # (N, L, D)
    dtype: np.dtype
    hidden_size: int
    num_states: int
    model_name: Optional[str] = None


def _infer_dtype(dtype_str: str) -> np.dtype:
    s = str(dtype_str).lower()
    if "float16" in s:
        return np.float16
    if "float32" in s:
        return np.float32
    raise ValueError(
        f"Unsupported dtype in meta.json: {dtype_str}. Use float16/float32 for numpy memmap.")


def load_cache_dir(cache_dir: str) -> Tuple[np.memmap, np.ndarray, CacheMeta, Dict]:
    meta_path = os.path.join(cache_dir, "meta.json")
    labels_path = os.path.join(cache_dir, "labels.npy")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json in {cache_dir}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels.npy in {cache_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    memmap_file = meta_raw.get(
        "memmap_file", "last_token_hidden_states.memmap")
    memmap_path = os.path.join(cache_dir, memmap_file)
    if not os.path.exists(memmap_path):
        raise FileNotFoundError(f"Missing memmap file: {memmap_path}")

    shape = tuple(meta_raw["shape"])  # (N, L, D)
    dtype = _infer_dtype(meta_raw["dtype"])
    hidden_size = int(meta_raw["hidden_size"])
    num_states = int(meta_raw["num_states"])
    model_name = meta_raw.get("model", None)

    feats = np.memmap(memmap_path, dtype=dtype, mode="r", shape=shape)
    labels = np.load(labels_path).astype(np.int64)

    if labels.shape[0] != shape[0]:
        raise ValueError(
            f"Label length mismatch: labels={labels.shape[0]} vs feats={shape[0]}")

    cm = CacheMeta(
        memmap_file=memmap_file,
        shape=shape,
        dtype=dtype,
        hidden_size=hidden_size,
        num_states=num_states,
        model_name=model_name,
    )
    return feats, labels, cm, meta_raw


# -----------------------------
# Geometric splits on indices
# -----------------------------
def geometric_split_indices(n: int, split_num: int = 5, ratio: float = 2.0, seed: int = 42) -> List[np.ndarray]:
    if split_num < 2:
        raise ValueError("split_num must be >= 2 for online MDL.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    denom = sum(ratio ** i for i in range(split_num))
    base = max(1, int(round(n / denom)))

    sizes = [max(1, int(round(base * (ratio ** i)))) for i in range(split_num)]
    total = sum(sizes)
    sizes[-1] += (n - total)

    if sizes[-1] <= 0:
        deficit = 1 - sizes[-1]
        sizes[-1] = 1
        i = split_num - 2
        while deficit > 0 and i >= 0:
            take = min(deficit, max(0, sizes[i] - 1))
            sizes[i] -= take
            deficit -= take
            i -= 1
        if deficit > 0:
            raise RuntimeError(
                "Could not construct valid geometric splits; dataset too small.")

    splits = []
    start = 0
    for s in sizes:
        splits.append(perm[start:start + s])
        start += s
    assert start == n
    return splits


# -----------------------------
# Probe builders
# -----------------------------
def build_probe(probe_type: str, d_in: int, hidden_sizes: List[int], num_classes: int = 2) -> Tuple[nn.Module, List[int]]:
    probe_type = probe_type.lower()
    if probe_type == "linear":
        return nn.Linear(d_in, num_classes), [d_in, num_classes]
    if probe_type in ("mlp", "multilinear"):
        assert hidden_sizes and len(
            hidden_sizes) > 0, "probe_hidden_size must be non-empty for mlp/multilinear"
        layer_dim = [d_in] + list(hidden_sizes) + [num_classes]
        if probe_type == "mlp":
            return MLP(layer_dim=layer_dim), layer_dim
        else:
            return MultiLinear(layer_dim=layer_dim), layer_dim
    raise ValueError(f"Unknown probe_type: {probe_type}")


# -----------------------------
# Tensor-only helpers (NO disk I/O inside training)
# -----------------------------
def make_batches(indices: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    """Return list of index tensors (views)"""
    return [indices[i:i + batch_size] for i in range(0, indices.numel(), batch_size)]


@torch.no_grad()
def ce_sum_on_indices(
    model: nn.Module,
    X: torch.Tensor,   # (N,D) on device
    y: torch.Tensor,   # (N,) on device
    idx: torch.Tensor,  # (M,) on device
    batch_size: int,
    desc: str,
    position: int,
) -> float:
    '''compute cross-entropy sum to get MDL'''
    model.eval()
    total = 0.0
    for b in tqdm(make_batches(idx, batch_size), desc=desc, position=position, leave=False):
        logits = model(X[b])
        total += float(F.cross_entropy(logits, y[b], reduction="sum").item())
    return total


@torch.no_grad()
def eval_acc_loss_on_indices(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    batch_size: int,
    desc: str,
    position: int,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for b in tqdm(make_batches(idx, batch_size), desc=desc, position=position, leave=False):
        logits = model(X[b])
        loss_sum += float(F.cross_entropy(logits,
                          y[b], reduction="sum").item())
        pred = logits.argmax(dim=-1)
        correct += int((pred == y[b]).sum().item())
        total += int(b.numel())
    return {"acc": correct / max(total, 1), "loss": loss_sum / max(total, 1), "n": float(total)}


def train_epoch_on_indices(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    batch_size: int,
    desc: str,
    position: int,
) -> float:
    '''train one epoch on given indices'''
    model.train()
    loss_sum = 0.0
    n = 0
    for b in tqdm(make_batches(idx, batch_size), desc=desc, position=position, leave=False):
        logits = model(X[b])
        loss = F.cross_entropy(logits, y[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_sum += float(loss.item()) * int(b.numel())
        n += int(b.numel())
    return loss_sum / max(n, 1)


# -----------------------------
# MDL stage training with early stop (tensor-only)
# -----------------------------
def train_stage_with_early_stop_mdl_tensor(
    probe: nn.Module,
    X: torch.Tensor,         # full train X for this layer, on device
    y: torch.Tensor,         # full train y, on device
    train_idx: torch.Tensor,  # indices for training set (growing), on device
    encode_idx: torch.Tensor,  # indices for encode set, on device
    *,
    batch_size: int,
    train_epochs: int,
    lr: float,
    early_gap: int,
    mdl_prefix: float,
    layer: int,
    stage: int,
    stages_total: int,
) -> Tuple[float, int]:
    """
    Early stopping by mdl
    Returns:
      best_ce_sum_for_stage, best_epoch
    """
    probe.to(X.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    best_total_mdl = float("inf")
    best_ce_sum = float("inf")
    best_epoch = 1
    epochs_since_best = 0

    # Shuffle training indices once per epoch (standard)
    for epoch in range(1, train_epochs + 1):
        perm = train_idx[torch.randperm(train_idx.numel(), device=X.device)]

        _avg_train_loss = train_epoch_on_indices(
            probe, opt, X, y, perm, batch_size,
            desc=f"Train (L{layer:02d} S{stage}/{stages_total}) ep {epoch}/{train_epochs}",
            position=2,
        )

        ce_sum = ce_sum_on_indices(
            probe, X, y, encode_idx, batch_size,
            desc=f"Encode CE (L{layer:02d} S{stage}/{stages_total})",
            position=3,
        )
        total_mdl = mdl_prefix + ce_sum

        if total_mdl < best_total_mdl - 1e-9:
            best_total_mdl = total_mdl
            best_ce_sum = ce_sum
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        log.info(
            f"[Layer {layer:02d} Stage {stage}/{stages_total}] "
            f"epoch={epoch} encode_ce_sum={ce_sum:.2f} total_mdl={total_mdl:.2f} "
            f"best_total_mdl={best_total_mdl:.2f} since_best={epochs_since_best}"
        )

        if early_gap > 0 and epochs_since_best >= early_gap:
            break

    return best_ce_sum, best_epoch


def online_mdl_for_layer_tensor(
    X: torch.Tensor,  # (N,D) on device
    y: torch.Tensor,  # (N,) on device
    splits: List[np.ndarray],
    cfg: DictConfig,
    *,
    layer: int,
) -> Tuple[float, int, List[int]]:
    '''iterate over splits for online MDL'''
    batch_size = int(cfg.mdl.batch_size)
    probe_type = str(cfg.mdl.probe_type)
    hidden_sizes = list(cfg.mdl.probe_hidden_size) if str(
        cfg.mdl.probe_type).lower() != "linear" else []
    train_epochs = int(cfg.mdl.train_epochs)
    early_gap = int(cfg.mdl.early_stopping_gap)
    lr = float(cfg.mdl.lr)

    d_in = X.shape[1]
    mdl_sum = 0.0
    total_encoded = 0
    stop_epochs: List[int] = []
    stages_total = len(splits) - 1

    # initial train idx
    train_idx_np = splits[0]

    stage_iter = tqdm(range(
        stages_total), desc=f"MDL stages (layer {layer:02d})", position=1, leave=False)
    for s in stage_iter:
        stage = s + 1
        encode_idx_np = splits[s + 1]

        # IMPORTANT: for encode CE, order doesn't matter -> sort for better locality if still on CPU;
        # here we're already on GPU, but sorting makes batching more cache-friendly anyway.
        encode_idx_np = np.sort(encode_idx_np)

        train_idx = torch.from_numpy(
            train_idx_np.astype(np.int64)).to(X.device)
        encode_idx = torch.from_numpy(
            encode_idx_np.astype(np.int64)).to(X.device)

        probe, _ = build_probe(probe_type, d_in=d_in,
                               hidden_sizes=hidden_sizes, num_classes=2)

        best_ce_sum, best_epoch = train_stage_with_early_stop_mdl_tensor(
            probe=probe,
            X=X,
            y=y,
            train_idx=train_idx,
            encode_idx=encode_idx,
            batch_size=batch_size,
            train_epochs=train_epochs,
            lr=lr,
            early_gap=early_gap,
            mdl_prefix=mdl_sum,
            layer=layer,
            stage=stage,
            stages_total=stages_total,
        )
        stop_epochs.append(best_epoch)

        mdl_sum += best_ce_sum
        total_encoded += int(encode_idx.numel())

        # grow training set
        train_idx_np = np.concatenate([train_idx_np, encode_idx_np], axis=0)

        stage_iter.set_postfix(
            mdl=float(mdl_sum), encoded=total_encoded, best_ep=best_epoch)

    return mdl_sum, total_encoded, stop_epochs


def train_final_probe_full_train_tensor(
    X_train: torch.Tensor, y_train: torch.Tensor,
    cfg: DictConfig,
    *,
    layer: int,
) -> nn.Module:
    '''build a new probe, train on full train set, for accuracy eval'''
    batch_size = int(cfg.mdl.batch_size)
    probe_type = str(cfg.mdl.probe_type)
    hidden_sizes = list(cfg.mdl.probe_hidden_size) if str(
        cfg.mdl.probe_type).lower() != "linear" else []
    train_epochs = int(cfg.mdl.train_epochs)
    early_gap = int(cfg.mdl.early_stopping_gap)
    lr = float(cfg.mdl.lr)

    d_in = X_train.shape[1]
    probe, _ = build_probe(probe_type, d_in=d_in,
                           hidden_sizes=hidden_sizes, num_classes=2)
    probe.to(X_train.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    all_idx = torch.arange(X_train.size(
        0), device=X_train.device, dtype=torch.long)

    best_loss = float("inf")
    best_state = None
    since_best = 0

    epoch_iter = tqdm(range(1, train_epochs + 1),
                      desc=f"Final train (layer {layer:02d})", position=1, leave=False)
    for epoch in epoch_iter:
        perm = all_idx[torch.randperm(all_idx.numel(), device=X_train.device)]
        avg_loss = train_epoch_on_indices(
            probe, opt, X_train, y_train, perm, batch_size,
            desc=f"Final batches (L{layer:02d}) ep {epoch}/{train_epochs}",
            position=2,
        )
        epoch_iter.set_postfix(avg_loss=float(avg_loss),
                               best=float(best_loss), since_best=since_best)

        if avg_loss < best_loss - 1e-9:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in probe.state_dict().items()}
            since_best = 0
        else:
            since_best += 1

        if early_gap > 0 and since_best >= early_gap:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)
        probe.to(X_train.device)
    return probe


# -----------------------------
# Hydra main
# -----------------------------
@hydra.main(version_base=None, config_path=".", config_name="config_cached")
def main(cfg: DictConfig):
    logging.getLogger().setLevel(logging.INFO)
    log.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    seed = int(cfg.mdl.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load caches (memmap + labels.npy)
    train_feats, train_labels_np, train_meta, train_meta_raw = load_cache_dir(
        cfg.dataset.train)
    test_feats, test_labels_np, test_meta, test_meta_raw = load_cache_dir(
        cfg.dataset.test)

    if train_meta.hidden_size != test_meta.hidden_size or train_meta.num_states != test_meta.num_states:
        raise ValueError(
            f"Train/Test mismatch: train(L={train_meta.num_states},D={train_meta.hidden_size}) "
            f"test(L={test_meta.num_states},D={test_meta.hidden_size})"
        )

    N_train, L, D = train_meta.shape
    N_test = test_meta.shape[0]
    log.info(
        f"Train cached feats: N={N_train} L={L} D={D} dtype={train_meta.dtype}")
    log.info(
        f"Test  cached feats: N={N_test} L={L} D={D} dtype={test_meta.dtype}")

    # Geometric splits on train indices (CPU arrays)
    split_num = 5
    ratio = 2.0
    splits = geometric_split_indices(
        N_train, split_num=split_num, ratio=ratio, seed=seed)
    log.info(f"Geometric splits sizes: {[len(s) for s in splits]}")

    # Labels to device once (small)
    y_train_all = torch.from_numpy(train_labels_np).to(device)
    y_test_all = torch.from_numpy(test_labels_np).to(device)

    # Results
    mdl_sum_by_layer: Dict[int, float] = {}
    nll_by_layer: Dict[int, float] = {}
    stop_epoch_by_layer: Dict[int, List[int]] = {}
    test_acc_by_layer: Dict[int, float] = {}
    test_loss_by_layer: Dict[int, float] = {}

    # Layer loop
    for layer in tqdm(range(L), desc="Layers", position=0, leave=True):
        # ---------
        # ONE-TIME disk touch per layer: load full layer matrix into torch
        # ---------
        # train_feats[:, layer, :] is a (N,D) view; np.asarray makes it contiguous in RAM
        X_train = torch.from_numpy(np.asarray(
            train_feats[:, layer, :], dtype=np.float32)).to(device)
        X_test = torch.from_numpy(np.asarray(
            test_feats[:, layer, :], dtype=np.float32)).to(device)

        # Online MDL (no disk reads now)
        log.info(f"\n[Layer {layer:02d}] Online MDL (tensor-only)...")
        mdl_sum, n_encoded, stop_epochs = online_mdl_for_layer_tensor(
            X=X_train,
            y=y_train_all,
            splits=splits,
            cfg=cfg,
            layer=layer,
        )
        mdl_sum_by_layer[layer] = float(mdl_sum)
        nll_by_layer[layer] = float(mdl_sum) / max(int(n_encoded), 1)
        stop_epoch_by_layer[layer] = stop_epochs

        log.info(
            f"[Layer {layer:02d}] total_MDL(sum CE)={mdl_sum:.2f}, encoded={n_encoded}, NLL={nll_by_layer[layer]:.4f}")
        log.info(
            f"[Layer {layer:02d}] stopping epochs per stage = {stop_epochs}")

        # Final train on full train (tensor-only) & evaluate on test (tensor-only)
        log.info(
            f"[Layer {layer:02d}] Final train + test eval (tensor-only)...")
        final_probe = train_final_probe_full_train_tensor(
            X_train, y_train_all, cfg, layer=layer)

        test_idx = torch.arange(N_test, device=device, dtype=torch.long)
        test_metrics = eval_acc_loss_on_indices(
            final_probe, X_test, y_test_all, test_idx,
            batch_size=int(cfg.mdl.batch_size),
            desc=f"Test eval (layer {layer:02d})",
            position=1,
        )
        test_acc_by_layer[layer] = float(test_metrics["acc"])
        test_loss_by_layer[layer] = float(test_metrics["loss"])
        log.info(
            f"[Layer {layer:02d}] test_acc={test_acc_by_layer[layer]:.4f} test_loss={test_loss_by_layer[layer]:.4f}")

        # Free GPU memory for next layer
        del X_train, X_test, final_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save results
    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "train_meta": train_meta_raw,
        "test_meta": test_meta_raw,
        "splits": [int(len(s)) for s in splits],
        "mdl_sum_ce": mdl_sum_by_layer,
        "mdl_nll": nll_by_layer,
        "stopping_epochs_per_stage": stop_epoch_by_layer,
        "test_acc": test_acc_by_layer,
        "test_loss": test_loss_by_layer,
        "probe_type": str(cfg.mdl.probe_type),
        "probe_hidden_size": list(cfg.mdl.probe_hidden_size) if str(cfg.mdl.probe_type).lower() != "linear" else [],
        "L": int(L),
        "D": int(D),
    }

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best_layer = min(mdl_sum_by_layer.keys(),
                     key=lambda k: mdl_sum_by_layer[k])
    log.info(
        f"\n[BEST by MDL] layer={best_layer} MDL={mdl_sum_by_layer[best_layer]:.2f} NLL={nll_by_layer[best_layer]:.4f}")
    log.info(f"[Outputs] {out_dir}")


if __name__ == "__main__":
    main()
