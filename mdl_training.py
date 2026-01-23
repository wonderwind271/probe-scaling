from mdl_utils import split_dataset  # and geometric_splits already in mdl_utils

from probe_model import FrozenBackboneLayerwiseProber, MLP, MultiLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import logging
import json
from collections import defaultdict
import os

log = logging.getLogger(__name__)


def merge_dicts_min(d1: dict, d2: dict) -> dict:
    merged = {}
    for k in d1.keys() | d2.keys():  # union of keys
        if k in d1 and k in d2:
            merged[k] = min(d1[k], d2[k])
        elif k in d1:
            merged[k] = d1[k]
        else:
            merged[k] = d2[k]

    return merged



def tokenize_fn(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def flatten_data(batch):
    '''Convert Seed42Lab/en-ud-test-pair into a unified format. clean: 1, corrupted: 0'''
    texts = []
    labels = []
    for pos, neg in zip(batch['positive'], batch['negative']):
        texts.append(pos)
        labels.append(1)

        texts.append(neg)
        labels.append(0)
    return {"text": texts, "label": labels}


def build_layerwise_probe_model(backbone, probe_type="mlp",
                                probe_hidden_size: list = [1024], label_num: int = 2):
    """
    Returns a fresh FrozenBackboneLayerwiseProber with new probe params.
    probe_type: "mlp", "multilinear" or "linear"
    """
    if hasattr(backbone.config, 'n_embd'):
        d = backbone.config.n_embd
    else:
        d = backbone.config.hidden_size

    if hasattr(backbone.config, 'n_layer'):
        n_layers = backbone.config.n_layer
    else:
        n_layers = backbone.config.num_hidden_layers

    if probe_type != 'linear':
        assert probe_hidden_size, 'probe_hidden_size must be non-empty for mlp/multilinear probes'
        layer_dim = [d]+probe_hidden_size+[label_num]

    if probe_type == "mlp":
        probes = {i: MLP(layer_dim=layer_dim) for i in range(n_layers + 1)}
    elif probe_type == "linear":
        probes = {i: nn.Linear(d, label_num) for i in range(n_layers + 1)}
    elif probe_type == "multilinear":
        probes = {i: MultiLinear(layer_dim=layer_dim)
                  for i in range(n_layers + 1)}
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    model = FrozenBackboneLayerwiseProber(
        backbone=backbone,
        probes=probes,
        pooling="last_token",
    )
    return model, layer_dim


@torch.no_grad()
def summed_ce_by_layer(model, data_loader, device):
    """
    MDL eval: summed CE per layer on a loader (UNSEEN CHUNK)
    Returns:
      ce_sum_by_layer: dict[layer -> float]  (sum over examples, NOT averaged)
      n_examples: int
    """
    model.eval()
    ce_sum_by_layer = {layer: 0.0 for layer in model.attached_layers()}
    n_examples = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)  # (B,)

        out = model(input_ids=input_ids,
                    attention_mask=attention_mask, labels=None)
        logits_by_layer = out["logits_by_layer"]  # dict[layer -> (B,2)]

        B = labels.size(0)
        n_examples += B

        for layer, logits in logits_by_layer.items():
            # sum reduction => total code length contribution for this layer over the batch
            loss_sum = F.cross_entropy(logits, labels, reduction="sum")
            ce_sum_by_layer[layer] += float(loss_sum.item())

    return ce_sum_by_layer, n_examples


def train_probe(model, train_loader, device, num_epochs=5, lr=1e-3):
    '''Supervised training on the current training chunk (train split)'''
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        total_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"  Train epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
            loss = out["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        logging.info(f"    avg_train_loss={avg_loss:.4f}")

    return model


def train_probe_w_eval(model, train_loader, test_loader, device, num_epochs=5, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    final_ce_sum_by_layer = {}
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"  Train epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
            loss = out["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        logging.info(f"    avg_train_loss={avg_loss:.4f}")
        ce_sum_by_layer, n_ex = summed_ce_by_layer(model, test_loader, device)
        final_ce_sum_by_layer = merge_dicts_min(final_ce_sum_by_layer, ce_sum_by_layer)

    return model, final_ce_sum_by_layer, n_ex


def evaluate_probe(tokenizer, cfg, model, device):
    '''Evaluate probe accuracy per layer on test set'''
    logging.info("\n[Probe Evaluation on Test Set]")
    test_set_orig = load_dataset(
        cfg.dataset.test, split=cfg.dataset.test_split)

    # clean: 1, corrupted: 0
    test_set = test_set_orig.map(
        flatten_data, batched=True,
        remove_columns=['positive', 'negative']
    ).shuffle()
    test_set = test_set.map(tokenize_fn, batched=True, remove_columns=[
                            "text"],  fn_kwargs={"tokenizer": tokenizer})
    test_set.set_format(type="torch", columns=[
                        "input_ids", "attention_mask", "label"])

    test_loader = DataLoader(
        test_set, batch_size=cfg.mdl.batch_size, shuffle=True)

    correct_num = defaultdict(int)
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        correct_by_layer = model(input_ids=input_ids, attention_mask=attention_mask,
                                 labels=labels, test=True)

        for layer, acc in correct_by_layer.items():
            correct_num[layer] += acc

    total_num = len(test_set)
    acc_by_layer = {}
    for layer, correct in correct_num.items():
        acc = correct / total_num
        acc_by_layer[layer] = acc
        logging.info(f"  Test accuracy - layer {layer}: {acc:.4f}")
    return acc_by_layer


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA not available. Running on CPU.")

    # tokenize and dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg.dataset.train, split=cfg.dataset.train_split)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=[
                          "text"],  fn_kwargs={"tokenizer": tokenizer})
    dataset.set_format(type="torch", columns=[
        "input_ids", "attention_mask", "label"])
    label_num = dataset.features["label"].num_classes
    logging.info(f'dataset has {len(dataset)} examples, {label_num} classes')

    # split into geometric chunks (already shuffles inside if seed>=0)
    splits = split_dataset(dataset=dataset, split_num=5,
                           ratio=2.0, seed=cfg.mdl.seed)
    for i, d in enumerate(splits):
        logging.debug(f"dataset D{i+1}: {len(d)}")

    # Online MDL loop
    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.model, device_map="cuda", torch_dtype=torch.float16)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # MDL accumulators per layer
    mdl_sum_by_layer = defaultdict(float)
    total_encoded_examples = 0
    train_data = splits[0]
    # training dataset grows: D1, D1+D2, ...
    for stage, (_, next_chunk) in enumerate(zip(splits, splits[1:])):
        logging.info(f"\n[MDL] Stage {stage+1}/{len(splits)-1}")
        logging.info(f"  Train on:  D1...D{stage+1} (n={len(train_data)})")
        logging.info(f"  Encode on: D{stage+2}      (n={len(next_chunk)})")

        # fresh probe each stage (paper-style, avoids warm-start artifacts)
        model, layer_dim = build_layerwise_probe_model(
            backbone=backbone, probe_type=cfg.mdl.probe_type,
            probe_hidden_size=cfg.mdl.probe_hidden_size, label_num=label_num)

        # loaders
        train_loader = DataLoader(
            train_data, batch_size=cfg.mdl.batch_size, shuffle=True)
        encode_loader = DataLoader(
            next_chunk, batch_size=cfg.mdl.batch_size, shuffle=False)

        # # train probe on current train_data
        # model = train_probe(model, train_loader, device,
        #                     num_epochs=cfg.mdl.train_epochs, lr=cfg.mdl.lr)

        # # compute summed CE on unseen chunk (this is the "code length" for this stage)
        # ce_sum_by_layer, n_ex = summed_ce_by_layer(
        #     model, encode_loader, device)
        model, ce_sum_by_layer, n_ex = train_probe_w_eval(model, train_loader, encode_loader, device, num_epochs=cfg.mdl.train_epochs, lr=cfg.mdl.lr)
        
        total_encoded_examples += n_ex
        for layer, ce_sum in ce_sum_by_layer.items():
            mdl_sum_by_layer[layer] += ce_sum        

        # (optional) print a compact summary: last layer + avg across layers
        layers = sorted(mdl_sum_by_layer.keys())
        last_layer = layers[-1]
        avg_stage_ce = sum(ce_sum_by_layer.values()) / len(ce_sum_by_layer)
        logging.info(
            f"  Stage CE sum (last_layer={last_layer}): {ce_sum_by_layer[last_layer]:.2f}")
        logging.info(f"  Stage CE avg over layers (sum): {avg_stage_ce:.2f}")

        # grow training set
        train_data = concatenate_datasets([train_data, next_chunk])

    layers = sorted(mdl_sum_by_layer.keys())
    last_layer = layers[-1]

    logging.info(f"Encoded examples total: {total_encoded_examples}")

    # Total MDL (sum CE) per layer
    logging.info("\n[Total MDL = summed CE over all encoded examples]")
    logging.info(
        f"Last layer ({last_layer}) MDL: {mdl_sum_by_layer[last_layer]:.2f}")
    logging.info(
        f"Avg over layers MDL: {sum(mdl_sum_by_layer.values())/len(mdl_sum_by_layer):.2f}")

    # Optional normalization: per-example NLL (NOT MDL proper, but useful for comparison)
    logging.info(
        "\n[Per-example NLL = MDL / #encoded_examples]  (useful normalized view)")
    logging.info(
        f"Last layer ({last_layer}) NLL: {mdl_sum_by_layer[last_layer] / total_encoded_examples:.4f}")
    logging.info(
        f"Avg over layers NLL: {(sum(mdl_sum_by_layer.values())/len(mdl_sum_by_layer)) / total_encoded_examples:.4f}")

    acc_by_layer = evaluate_probe(tokenizer, cfg, model, device)

    output_dir = HydraConfig.get().runtime.output_dir
    with open(os.path.join(output_dir, 'results.json'), "w") as f:
        json.dump({'mdl': mdl_sum_by_layer,
                   'acc': acc_by_layer,
                   'probe_size': list(layer_dim),
                   'probe_type': cfg.mdl.probe_type,
                   'base_model_name': cfg.model},
                  f, indent=2)


if __name__ == "__main__":
    main()
