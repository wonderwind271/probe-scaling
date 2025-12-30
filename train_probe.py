from probe_model import FrozenBackboneLayerwiseProber, MLP  # make sure MLP is available here
from datasets import load_dataset, concatenate_datasets

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import GPT2TokenizerFast, GPT2LMHeadModel


# -------------------------
# Device setup (do this first)
# -------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[WARNING] CUDA not available. Running on CPU.")


# -------------------------
# Tokenizer (define before tokenize_fn)
# -------------------------
model_path = 'gpt2_delta_scrub_L2-4/merged_full'

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",   # or "longest"
        max_length=128,
    )


# -------------------------
# Model
# -------------------------
backbone = GPT2LMHeadModel.from_pretrained(model_path)
d = backbone.config.n_embd

probes = {i: MLP(d, d, 2) for i in range(backbone.config.n_layer + 1)}

model = FrozenBackboneLayerwiseProber(
    backbone=backbone,
    probes=probes,
    pooling="last_token",
)

model = model.to(device)


# -------------------------
# Datasets (train + test)
# -------------------------
seed = 42

dataset1 = load_dataset("Seed42Lab/en_gum-ud-train", split="train")
dataset2 = load_dataset("Seed42Lab/en_ewt-ud-train", split="train")
dataset = concatenate_datasets([dataset1, dataset2]).shuffle(seed=seed)

# dataset.push_to_hub("Seed42Lab/en-ud-train")

dataset_test1 = load_dataset("Seed42Lab/en_gum-ud-test", split="train")
dataset_test2 = load_dataset("Seed42Lab/en_ewt-ud-test", split="train")
dataset_test = concatenate_datasets([dataset_test1, dataset_test2]) 
# dataset_test.push_to_hub("Seed42Lab/en-ud-test")


# Tokenize + set torch format
dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"],
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

dataset_test = dataset_test.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"],
)
dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# -------------------------
# DataLoaders
# -------------------------
batch_size = 16

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


# -------------------------
# Optimizer
# -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# -------------------------
# Eval helper (accuracy per layer + optional average)
# -------------------------
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct_by_layer = {layer: 0 for layer in model.attached_layers()}
    total = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        total += labels.size(0)

        for layer, logits in out["logits_by_layer"].items():
            preds = logits.argmax(dim=-1)
            correct_by_layer[layer] += (preds == labels).sum().item()

    acc_by_layer = {layer: correct / total for layer, correct in correct_by_layer.items()}
    # also report average accuracy across layers
    avg_acc = sum(acc_by_layer.values()) / len(acc_by_layer)
    return acc_by_layer, avg_acc


# -------------------------
# Training loop
# -------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = out["loss"]
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # update bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    # Evaluate on test set
    acc_by_layer, avg_acc = evaluate(model, test_loader, device)

    # Print a compact summary (last layer + avg)
    last_layer = max(acc_by_layer.keys())
    print(
        f"Epoch {epoch+1}: train_loss={avg_loss:.4f} | "
        f"test_acc(last_layer={last_layer})={acc_by_layer[last_layer]:.4f} | "
        f"test_acc(avg_layers)={avg_acc:.4f}"
    )
    print(acc_by_layer)
