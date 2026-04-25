from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_bias(linear: nn.Linear) -> None:
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_gamma(linear: nn.Linear, init_gamma: float) -> None:
    std = float(linear.in_features) ** (-float(init_gamma))
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    _zero_bias(linear)


def _init_zero(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    _zero_bias(linear)


def _init_ntk_or_mf(linear1: nn.Linear, linear2: nn.Linear, d_in: int) -> None:
    '''Scale: N(0,1)'''
    std_w1 = 1.0 / math.sqrt(d_in)
    nn.init.normal_(linear1.weight, mean=0.0, std=std_w1)
    _zero_bias(linear1)
    nn.init.normal_(linear2.weight, mean=0.0, std=1.0)
    _zero_bias(linear2)


def _init_mup(linear1: nn.Linear, linear2: nn.Linear, d_in: int, width: int) -> None:
    std_w1 = 1.0 / math.sqrt(d_in)
    std_w2 = 1.0 / float(width)
    nn.init.normal_(linear1.weight, mean=0.0, std=std_w1)
    _zero_bias(linear1)
    nn.init.normal_(linear2.weight, mean=0.0, std=std_w2)
    _zero_bias(linear2)


def get_lr_rule(strategy: str) -> str:
    strategy = strategy.lower()
    if strategy == 'ntk':
        return 'base_lr'
    if strategy == 'mf':
        return 'base_lr*width'
    if strategy == 'mup':
        return 'hidden=base_lr,readout=base_lr*(1/width)'
    return 'base_lr'


def get_bias_rule(strategy: str) -> str:
    if strategy.lower() == 'standard':
        return 'pytorch_default'
    return 'zero'


def get_forward_prefactor(strategy: str, width: int) -> float:
    strategy = strategy.lower()
    if strategy == 'ntk':
        return 1.0 / math.sqrt(width)
    if strategy == 'mf':
        return 1.0 / float(width)
    return 1.0


def get_effective_lr(base_lr: float, strategy: str, width: int) -> object:
    strategy = strategy.lower()
    if strategy == 'ntk':
        return float(base_lr)
    if strategy == 'mf':
        return float(base_lr * width)
    if strategy == 'mup':
        readout_lr = float(base_lr / width)
        return {'hidden': float(base_lr), 'readout': readout_lr}
    return float(base_lr)


class InitStudyMLPProbe(nn.Module):
    """Two-layer probe used in the initialization study."""

    def __init__(self, d_in: int, hidden_width: int, num_classes: int = 2,
                 init_strategy: str = 'standard',
                 init_gamma: Optional[float] = None):
        super().__init__()
        self.d_in = int(d_in)
        self.hidden_width = int(hidden_width)
        self.num_classes = int(num_classes)
        self.init_strategy = str(init_strategy).lower()
        self.init_gamma = init_gamma
        self.layer1 = nn.Linear(self.d_in, self.hidden_width)
        self.layer2 = nn.Linear(self.hidden_width, self.num_classes)
        self.output_scale = get_forward_prefactor(
            self.init_strategy, self.hidden_width)
        self._init_layers()

    def _init_layers(self) -> None:
        strategy = self.init_strategy
        if strategy == 'standard':
            return
        if strategy == 'gamma':
            self._require_gamma()
            _init_gamma(self.layer1, self.init_gamma)
            _init_gamma(self.layer2, self.init_gamma)
            return
        if strategy == 'gamma_first_zero':
            self._require_gamma()
            _init_zero(self.layer1)
            _init_gamma(self.layer2, self.init_gamma)
            return
        if strategy == 'gamma_second_zero':
            self._require_gamma()
            _init_gamma(self.layer1, self.init_gamma)
            _init_zero(self.layer2)
            return
        if strategy in ('ntk', 'mf'):
            _init_ntk_or_mf(self.layer1, self.layer2, self.d_in)
            return
        if strategy == 'mup':
            _init_mup(self.layer1, self.layer2, self.d_in, self.hidden_width)
            return
        raise ValueError(f'Unknown init strategy: {self.init_strategy}')

    def _require_gamma(self) -> None:
        if self.init_gamma is None:
            raise ValueError('Gamma-based strategies require init_gamma.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x * self.output_scale


def build_init_study_probe(d_in: int, hidden_width: int, num_classes: int,
                           init_strategy: str,
                           init_gamma: Optional[float]) -> InitStudyMLPProbe:
    return InitStudyMLPProbe(
        d_in=d_in,
        hidden_width=hidden_width,
        num_classes=num_classes,
        init_strategy=init_strategy,
        init_gamma=init_gamma,
    )


def _build_optimizer_cls(optimizer_name: str):
    name = optimizer_name.lower()
    if name == 'adamw':
        return torch.optim.AdamW
    if name == 'adam':
        return torch.optim.Adam
    if name == 'sgd':
        return torch.optim.SGD
    raise ValueError(
        f'Unsupported optimizer {optimizer_name}. '
        'Expected adamw, adam, or sgd.')


def build_init_study_optimizer(probe: InitStudyMLPProbe,
                               optimizer_name: str,
                               base_lr: float
                               ) -> Tuple[torch.optim.Optimizer, object, str]:
    opt_cls = _build_optimizer_cls(optimizer_name)
    width = probe.hidden_width
    strategy = probe.init_strategy
    lr_rule = get_lr_rule(strategy)
    effective_lr = get_effective_lr(base_lr, strategy, width)
    if strategy == 'mup':
        param_groups = [
            {
                'params': [probe.layer1.weight, probe.layer1.bias, probe.layer2.bias],
                'lr': float(effective_lr['hidden']),
            },
            {
                'params': [probe.layer2.weight],
                'lr': float(effective_lr['readout']),
            },
        ]
        return opt_cls(param_groups), effective_lr, lr_rule
    return opt_cls(probe.parameters(), lr=float(effective_lr)), effective_lr, lr_rule


def clone_probe_state(probe: InitStudyMLPProbe) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in probe.state_dict().items()
    }


def flatten_state_diff(cur_state: Dict[str, torch.Tensor],
                       init_state: Dict[str, torch.Tensor],
                       keys: Optional[Tuple[str, ...]] = None) -> float:
    vec = []
    state_keys = keys if keys is not None else tuple(cur_state.keys())
    for key in state_keys:
        diff = cur_state[key].detach().cpu().reshape(-1) - init_state[key].reshape(-1)
        vec.append(diff)
    if not vec:
        return 0.0
    return float(torch.cat(vec).norm(p=2).item())


def compute_probe_drift(probe: InitStudyMLPProbe,
                        init_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    cur_state = {
        name: tensor.detach().cpu()
        for name, tensor in probe.state_dict().items()
    }
    return {
        'total': flatten_state_diff(cur_state, init_state),
        'layer0': flatten_state_diff(
            cur_state,
            init_state,
            keys=('layer1.weight', 'layer1.bias'),
        ),
        'layer1': flatten_state_diff(
            cur_state,
            init_state,
            keys=('layer2.weight', 'layer2.bias'),
        ),
    }
