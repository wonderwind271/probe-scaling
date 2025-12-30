from typing import Dict, Optional, Literal, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


PoolType = Literal["last_token", "mean"]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def masked_mean(x, mask):
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

def pool_hidden(h, attention_mask: Optional[torch.Tensor], pool: PoolType):
    # h: (B,T,D)
    if pool == "mean":
        return h.mean(dim=1) if attention_mask is None else masked_mean(h, attention_mask)
    if pool == "last_token":
        if attention_mask is None:
            return h[:, -1, :]
        idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        return h[torch.arange(h.size(0), device=h.device), idx, :]
    raise ValueError(pool)

class FrozenBackboneLayerwiseProber(nn.Module):
    """
    Frozen backbone + layerwise probes.
    - labels optional: if labels is None, returns logits only (no loss).
    - backbone can be GPT2Model or GPT2LMHeadModel (anything that returns hidden_states when asked).
    """

    def __init__(
        self,
        backbone: nn.Module,
        probes: Dict[int, nn.Module],
        pooling: PoolType = "last_token",
        freeze_backbone: bool = True,
        backbone_no_grad: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.backbone_no_grad = backbone_no_grad

        self.probes = nn.ModuleDict({f"layer_{k}": v for k, v in probes.items()})

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def attached_layers(self):
        return sorted(int(k.split("_")[1]) for k in self.probes.keys())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,   # (B,) for standard classification
        loss_fn=None,                            # optional custom loss
        **kwargs,
    ):
        # Run backbone (usually no_grad to save memory)
        ctx = torch.no_grad() if self.backbone_no_grad else torch.enable_grad()
        with ctx:
            outputs = self.backbone.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            hidden_states = outputs.hidden_states  # tuple (n_layers+1), each (B,T,D)

        logits_by_layer = {}
        for layer_idx in self.attached_layers():
            h = hidden_states[layer_idx]  # (B,T,D)
            feat = pool_hidden(h, attention_mask, self.pooling)  # (B,D)
            logits_by_layer[layer_idx] = self.probes[f"layer_{layer_idx}"](feat)

        loss = None
        if labels is not None:
            per_layer_losses = []
            for logits in logits_by_layer.values():
                if loss_fn is None:
                    per_layer_losses.append(F.cross_entropy(logits, labels))
                else:
                    per_layer_losses.append(loss_fn(logits, labels))
            loss = torch.stack(per_layer_losses).mean()

        return {
            "logits_by_layer": logits_by_layer,  # layer -> (B,C) or whatever your probe returns
            "loss": loss,                        # None if labels is None
        }


