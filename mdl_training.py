from mdl_utils import split_dataset  # and geometric_splits already in mdl_utils

from probe_model import FrozenBackboneLayerwiseProber, MLP, MultiLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Device setup
# -------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[WARNING] CUDA not available. Running on CPU.")


# -------------------------
# Tokenizer
# -------------------------
model_path = "meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


# -------------------------
# Build model (fresh each stage!)
# -------------------------
def build_layerwise_probe_model(backbone, probe_type="mlp"):
    """
    Returns a fresh FrozenBackboneLayerwiseProber with new probe params.
    probe_type: "mlp" or "linear"
    """
    if hasattr(backbone.config, 'n_embd'):
        d = backbone.config.n_embd
    else:
        d = backbone.config.hidden_size

    if hasattr(backbone.config, 'n_layer'):
        n_layers = backbone.config.n_layer
    else:
        n_layers = backbone.config.num_hidden_layers

    if probe_type == "mlp":
        probes = {i: MLP(d, int(d/8), 2) for i in range(n_layers + 1)}
    elif probe_type == "linear":
        probes = {i: nn.Linear(d, 2) for i in range(n_layers + 1)}
    elif probe_type == "multilinear":
        probes = {i: MultiLinear(d, int(d/4), 2) for i in range(n_layers + 1)}
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    model = FrozenBackboneLayerwiseProber(
        backbone=backbone,
        probes=probes,
        pooling="last_token",
    )
    return model


# -------------------------
# MDL eval: summed CE per layer on a loader (UNSEEN CHUNK)
# -------------------------
@torch.no_grad()
def summed_ce_by_layer(model, data_loader, device):
    """
    Returns:
      ce_sum_by_layer: dict[layer -> float]  (sum over examples, NOT averaged)
      n_examples: int
    """
    model.eval()
    ce_sum_by_layer = {layer: 0.0 for layer in model.attached_layers()}
    n_examples = 0
    print_state = 1
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)  # (B,)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        # if print_state == 1:
        #     print(input_ids,attention_mask)
        #     print_state = 0
        logits_by_layer = out["logits_by_layer"]  # dict[layer -> (B,2)]

        B = labels.size(0)
        n_examples += B
        
        for layer, logits in logits_by_layer.items():
            # sum reduction => total code length contribution for this layer over the batch
            
            loss_sum = F.cross_entropy(logits, labels, reduction="sum")
            ce_sum_by_layer[layer] += float(loss_sum.item())

    return ce_sum_by_layer, n_examples


# -------------------------
# Supervised training on the current training chunk (train split)
# -------------------------
def train_probe(model, train_loader, device, num_epochs=5, lr=1e-3):
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # print_state = 1
    for epoch in range(num_epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"  Train epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # if print_state == 1:
            #     print(out)
            #     print_state = 0

            loss = out["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        # (optional) print per epoch
        # print(f"    avg_train_loss={avg_loss:.4f}")

    return model


# -------------------------
# Prepare dataset (single dataset; no test)
# -------------------------
seed = 42
dataset = load_dataset("Seed42Lab/en-ud-train", split="train")

# tokenize + torch format
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# split into geometric chunks (already shuffles inside if seed>=0)
splits = split_dataset(dataset=dataset, split_num=5, ratio=2.0, seed=seed)
for i, d in enumerate(splits):
    print(f"D{i+1}: {len(d)}")


# -------------------------
# Online MDL loop
# -------------------------
batch_size = 16
probe_type = "mlp"      # "mlp", "linear"
train_epochs = 1
lr = 1e-3

# Load backbone once (frozen)
backbone = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda",torch_dtype=torch.float16)
backbone = backbone.to(device)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad_(False)

# MDL accumulators per layer
# We'll create them lazily once we know layers
mdl_sum_by_layer = None
total_encoded_examples = 0

# training dataset grows: D1, D1+D2, ...
train_data = splits[0]

for stage in range(len(splits) - 1):
    next_chunk = splits[stage + 1]

    print(f"\n[MDL] Stage {stage+1}/{len(splits)-1}")
    print(f"  Train on:  D1..D{stage+1} (n={len(train_data)})")
    print(f"  Encode on: D{stage+2}      (n={len(next_chunk)})")

    # fresh probe each stage (paper-style, avoids warm-start artifacts)
    model = build_layerwise_probe_model(backbone=backbone, probe_type=probe_type)

    # loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    encode_loader = DataLoader(next_chunk, batch_size=batch_size, shuffle=False)

    # train probe on current train_data
    model = train_probe(model, train_loader, device, num_epochs=train_epochs, lr=lr)

    # compute summed CE on unseen chunk (this is the "code length" for this stage)
    ce_sum_by_layer, n_ex = summed_ce_by_layer(model, encode_loader, device)
    total_encoded_examples += n_ex

    if mdl_sum_by_layer is None:
        mdl_sum_by_layer = {layer: 0.0 for layer in ce_sum_by_layer.keys()}

    for layer, ce_sum in ce_sum_by_layer.items():
        mdl_sum_by_layer[layer] += ce_sum

    # (optional) print a compact summary: last layer + avg across layers
    layers = sorted(mdl_sum_by_layer.keys())
    last_layer = layers[-1]
    avg_stage_ce = sum(ce_sum_by_layer.values()) / len(ce_sum_by_layer)
    print(f"  Stage CE sum (last_layer={last_layer}): {ce_sum_by_layer[last_layer]:.2f}")
    print(f"  Stage CE avg over layers (sum): {avg_stage_ce:.2f}")

    # grow training set
    train_data = concatenate_datasets([train_data, next_chunk])


# -------------------------
# Final reporting
# -------------------------
layers = sorted(mdl_sum_by_layer.keys())
last_layer = layers[-1]

print("\n========== Online MDL Results ==========")
print(f"Encoded examples total: {total_encoded_examples}")

# Total MDL (sum CE) per layer
print("\n[Total MDL = summed CE over all encoded examples]")
print(f"Last layer ({last_layer}) MDL: {mdl_sum_by_layer[last_layer]:.2f}")
print(f"Avg over layers MDL: {sum(mdl_sum_by_layer.values())/len(mdl_sum_by_layer):.2f}")

# Optional normalization: per-example NLL (NOT MDL proper, but useful for comparison)
print("\n[Per-example NLL = MDL / #encoded_examples]  (useful normalized view)")
print(f"Last layer ({last_layer}) NLL: {mdl_sum_by_layer[last_layer] / total_encoded_examples:.4f}")
print(f"Avg over layers NLL: {(sum(mdl_sum_by_layer.values())/len(mdl_sum_by_layer)) / total_encoded_examples:.4f}")

# If you want full dict:
print(mdl_sum_by_layer)
