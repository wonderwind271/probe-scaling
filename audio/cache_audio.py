"""
Use this to create data folder capatible to mdl_training_cached.py
"""

import os
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_dataset, Audio
from transformers import WhisperModel, WhisperProcessor

# -----------------------------
# Config
# -----------------------------


MODEL_NAME = "openai/whisper-base"
DATASET_NAME = "wonderwind271/speech-dataset"
SAVE_DIR = "/scratch/chaijy_root/chaijy2/shuyuwu/whisper_hidden_states/train_mean_labeled"
DTYPE = torch.float16   # save space
DEVICE = "cuda"

# -----------------------------
# Load model / processor / data
# -----------------------------
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

ds = load_dataset(DATASET_NAME)
ds_train = ds["train"]

# Optional but recommended: force audio to 16kHz
ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))

save_dir = Path(SAVE_DIR)
save_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Extraction loop
# -----------------------------
with torch.no_grad():
    for idx, ex in enumerate(tqdm(ds_train, desc="Extracting train hidden states")):
        out_path = save_dir / f"{idx:08d}.pt"

        if out_path.exists():
            continue

        audio = ex["audio"]
        language = ex['language']
        # Processor output: [1, 80, 3000]
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features.to(DEVICE)

        # Encoder hidden states: tuple of length 7 for whisper-base
        encoder_outputs = model.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = encoder_outputs.hidden_states

        # Move to CPU and optionally cast
        hidden_states_cpu = tuple(h.squeeze(0).mean(dim=0).to("cpu", dtype=DTYPE) for h in hidden_states)
        # each tensor is now [512]

        save_obj = {
            "idx": idx,
            "hidden_states": hidden_states_cpu,
            "language": language,
            "audio_num_samples": len(audio["array"]),
            "audio_sampling_rate": audio["sampling_rate"],
        }

        torch.save(save_obj, out_path)