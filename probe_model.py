from __future__ import annotations

from typing import Dict, Optional, Literal, List
import torch
import torch.nn as nn
import torch.nn.functional as F

PoolType = Literal["last_token", "mean"]


class MLP(nn.Module):
    def __init__(self, layer_dim: List[int]):
        super().__init__()
        assert len(layer_dim) >= 3, 'MLP must have at least one hidden layer!'
        self.fc = nn.ModuleList()
        for (dim_in, dim_out) in zip(layer_dim, layer_dim[1:]):
            self.fc.append(nn.Linear(dim_in, dim_out))

    def forward(self, x):
        for idx, fc in enumerate(self.fc):
            if idx != len(self.fc)-1:
                x = F.relu(fc(x))
            else:
                x = fc(x)
        return x


class MultiLinear(nn.Module):
    def __init__(self, layer_dim: List[int]):
        super().__init__()
        assert len(
            layer_dim) >= 3, 'MultiLinear must have at least one hidden layer!'
        self.fc = nn.ModuleList()
        for (dim_in, dim_out) in zip(layer_dim, layer_dim[1:]):
            self.fc.append(nn.Linear(dim_in, dim_out))

    def forward(self, x):
        for fc in self.fc:
            x = fc(x)
        return x


def masked_mean(x, mask):
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def pool_hidden(h, attention_mask: Optional[torch.Tensor], pool: PoolType):
    # h: (B,T,D)
    if pool == "mean":
        return h.mean(dim=1) if attention_mask is None else masked_mean(h, attention_mask)
    elif pool == "last_token":
        if attention_mask is None:
            return h[:, -1, :]
        idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        return h[torch.arange(h.size(0), device=h.device), idx, :]
    else:
        raise ValueError(f'{pool} is not supported')


def _get_backbone_core(backbone: nn.Module) -> nn.Module:
    """
    Return the transformer "core" model that yields hidden_states, regardless of
    whether `backbone` is a *ForCausalLM wrapper or a base model.
    Works for common families:
      - Llama/Mistral/Qwen: `.base_model` / `.model`
      - GPT-2/GPT-J/GPT-Neo: `.base_model` / `.transformer`
      - GPT-NeoX (Pythia): `.base_model` / `.gpt_neox`
    """
    core = getattr(backbone, "base_model", None)
    if isinstance(core, nn.Module):
        return core

    for attr in ("model", "transformer", "gpt_neox", "backbone"):
        candidate = getattr(backbone, attr, None)
        if isinstance(candidate, nn.Module):
            return candidate

    return backbone


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

        self.probes = nn.ModuleDict(
            {f"layer_{k}": v for k, v in probes.items()})

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def attached_layers(self):
        return sorted(int(k.split("_")[1]) for k in self.probes.keys())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # (B,) for standard classification
        labels: Optional[torch.Tensor] = None,
        loss_fn=None,                            # optional custom loss
        test=False,
        **kwargs,
    ):
        # Run backbone (usually no_grad to save memory)
        ctx = torch.no_grad() if self.backbone_no_grad else torch.enable_grad()
        with ctx:
            backbone_core = _get_backbone_core(self.backbone)
            outputs = backbone_core(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            # tuple (n_layers+1), each (B,T,D)
            hidden_states = outputs.hidden_states

        logits_by_layer = {}  # (B, #num_class)
        for layer_idx in self.attached_layers():
            h = hidden_states[layer_idx]  # (B,T,D)
            feat = pool_hidden(h, attention_mask, self.pooling)  # (B,D)
            feat = feat.float()
            logits_by_layer[layer_idx] = self.probes[f"layer_{layer_idx}"](
                feat)

        if not test:
            loss = None
            if labels is not None:
                per_layer_losses = []
                for logits in logits_by_layer.values():
                    if loss_fn is None:
                        per_layer_losses.append(
                            F.cross_entropy(logits, labels))
                    else:
                        per_layer_losses.append(loss_fn(logits, labels))
                # avg over layers
                loss = torch.stack(per_layer_losses).mean()
            return {
                # layer -> (B,C) or whatever your probe returns
                "logits_by_layer": logits_by_layer,
                "loss": loss,                        # None if labels is None
            }

        else:  # only return acc
            correct_by_layer = {}
            for layer, logits in logits_by_layer.items():
                preds = logits.argmax(dim=-1)
                correct_num = (preds == labels).sum().item()
                correct_by_layer[layer] = correct_num
            return correct_by_layer
