from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from mdl_training_cached import ce_sum_on_indices, eval_acc_loss_on_indices, geometric_split_indices, load_cache_dir, train_epoch_on_indices
from probe_init_study import build_init_study_optimizer, build_init_study_probe, clone_probe_state, compute_probe_drift, get_bias_rule, get_effective_lr, get_forward_prefactor, get_lr_rule

log = logging.getLogger(__name__)


def _gamma_tag(init_gamma: Optional[float]) -> str:
    if init_gamma is None:
        return 'na'
    return str(init_gamma)


def get_custom_dir(model_short: str, task_name: str, probe_hidden_size: List[int],
                   seed: int, init_strategy: str,
                   init_gamma: Optional[float], optimizer: str) -> str:
    hidden_size = '-'.join(map(str, probe_hidden_size))
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    gamma_tag = _gamma_tag(init_gamma)
    return (
        f'init_results/{model_short}/{task_name}/hidden-{hidden_size}/'
        f'seed-{seed}/strategy-{init_strategy}/gamma-{gamma_tag}/opt-{optimizer}/{cur_time}'
    )


def validate_cfg(cfg: DictConfig) -> None:
    if str(cfg.mdl.probe_type).lower() != 'mlp':
        raise ValueError('The init study runner only supports probe_type=mlp.')
    if len(list(cfg.mdl.probe_hidden_size)) != 1:
        raise ValueError(
            'The init study runner only supports one hidden layer width.'
        )
    strategy = str(cfg.mdl.init_strategy).lower()
    gamma_strategies = {'gamma', 'gamma_first_zero', 'gamma_second_zero'}
    valid_strategies = gamma_strategies | {'standard', 'ntk', 'mf', 'mup'}
    if strategy not in valid_strategies:
        raise ValueError(f'Unknown init strategy: {strategy}')
    if strategy in gamma_strategies and cfg.mdl.get('init_gamma', None) is None:
        raise ValueError('Gamma-based strategies require mdl.init_gamma.')


def build_probe(cfg: DictConfig, d_in: int) -> nn.Module:
    hidden_width = int(list(cfg.mdl.probe_hidden_size)[0])
    init_gamma = cfg.mdl.get('init_gamma', None)
    if init_gamma is not None:
        init_gamma = float(init_gamma)
    return build_init_study_probe(
        d_in=d_in,
        hidden_width=hidden_width,
        num_classes=int(cfg.mdl.num_classes),
        init_strategy=str(cfg.mdl.init_strategy),
        init_gamma=init_gamma,
    )


def build_optimizer(cfg: DictConfig, probe: nn.Module
                    ) -> Tuple[torch.optim.Optimizer, object, str]:
    return build_init_study_optimizer(
        probe=probe,
        optimizer_name=str(cfg.mdl.optimizer),
        base_lr=float(cfg.mdl.lr),
    )


def train_stage_with_early_stop_mdl_tensor(probe: nn.Module, X: torch.Tensor,
                                           y: torch.Tensor,
                                           train_idx: torch.Tensor,
                                           encode_idx: torch.Tensor, *,
                                           batch_size: int,
                                           train_epochs: int,
                                           early_gap: int,
                                           mdl_prefix: float,
                                           layer: int,
                                           stage: int,
                                           stages_total: int,
                                           opt: torch.optim.Optimizer
                                           ) -> Tuple[float, int]:
    best_total_mdl = float('inf')
    best_ce_sum = float('inf')
    best_epoch = 1
    since_best = 0

    for epoch in range(1, train_epochs + 1):
        perm = train_idx[torch.randperm(train_idx.numel(), device=X.device)]
        train_epoch_on_indices(
            probe,
            opt,
            X,
            y,
            perm,
            batch_size,
            desc=f'Train (L{layer:02d} S{stage}/{stages_total}) ep {epoch}/{train_epochs}',
            position=2,
        )
        ce_sum = ce_sum_on_indices(
            probe,
            X,
            y,
            encode_idx,
            batch_size,
            desc=f'Encode CE (L{layer:02d} S{stage}/{stages_total})',
            position=3,
        )
        total_mdl = mdl_prefix + ce_sum
        if total_mdl < best_total_mdl - 1e-9:
            best_total_mdl = total_mdl
            best_ce_sum = ce_sum
            best_epoch = epoch
            since_best = 0
        else:
            since_best += 1

        log.info(
            f'[Layer {layer:02d} Stage {stage}/{stages_total}] '
            f'epoch={epoch} encode_ce_sum={ce_sum:.2f} '
            f'total_mdl={total_mdl:.2f} best_total_mdl={best_total_mdl:.2f} '
            f'since_best={since_best}'
        )
        if early_gap > 0 and since_best >= early_gap:
            break

    return best_ce_sum, best_epoch


def online_mdl_for_layer_tensor(X: torch.Tensor, y: torch.Tensor,
                                splits: List[np.ndarray], cfg: DictConfig, *,
                                layer: int) -> Tuple[float, int, List[int]]:
    batch_size = int(cfg.mdl.batch_size)
    train_epochs = int(cfg.mdl.train_epochs)
    early_gap = int(cfg.mdl.early_stopping_gap)
    stages_total = len(splits) - 1
    train_idx_np = splits[0]
    d_in = int(X.shape[1])
    mdl_sum = 0.0
    total_encoded = 0
    stop_epochs: List[int] = []

    stage_iter = tqdm(
        range(stages_total),
        desc=f'MDL stages (layer {layer:02d})',
        position=1,
        leave=False,
    )
    for stage_idx in stage_iter:
        stage = stage_idx + 1
        encode_idx_np = np.sort(splits[stage_idx + 1])
        train_idx = torch.from_numpy(train_idx_np.astype(np.int64)).to(X.device)
        encode_idx = torch.from_numpy(encode_idx_np.astype(np.int64)).to(X.device)
        probe = build_probe(cfg, d_in=d_in).to(X.device)
        opt, _, _ = build_optimizer(cfg, probe)
        best_ce_sum, best_epoch = train_stage_with_early_stop_mdl_tensor(
            probe=probe,
            X=X,
            y=y,
            train_idx=train_idx,
            encode_idx=encode_idx,
            batch_size=batch_size,
            train_epochs=train_epochs,
            early_gap=early_gap,
            mdl_prefix=mdl_sum,
            layer=layer,
            stage=stage,
            stages_total=stages_total,
            opt=opt,
        )
        stop_epochs.append(best_epoch)
        mdl_sum += best_ce_sum
        total_encoded += int(encode_idx.numel())
        train_idx_np = np.concatenate([train_idx_np, encode_idx_np], axis=0)
        stage_iter.set_postfix(
            mdl=float(mdl_sum),
            encoded=total_encoded,
            best_ep=best_epoch,
        )
    return mdl_sum, total_encoded, stop_epochs


def train_final_probe_full_train_tensor(X_train: torch.Tensor, y_train: torch.Tensor, cfg: DictConfig, *, layer: int) -> Tuple[nn.Module, float, float]:
    batch_size = int(cfg.mdl.batch_size)
    train_epochs = int(cfg.mdl.train_epochs)
    early_gap = int(cfg.mdl.early_stopping_gap)
    d_in = int(X_train.shape[1])
    probe = build_probe(cfg, d_in=d_in).to(X_train.device)
    opt, _, _ = build_optimizer(cfg, probe)
    init_state = clone_probe_state(probe)
    all_idx = torch.arange(X_train.size(0), device=X_train.device, dtype=torch.long)
    best_loss = float('inf')
    best_state = None
    since_best = 0

    epoch_iter = tqdm(
        range(1, train_epochs + 1),
        desc=f'Final train (layer {layer:02d})',
        position=1,
        leave=False,
    )
    for epoch in epoch_iter:
        perm = all_idx[torch.randperm(all_idx.numel(), device=X_train.device)]
        avg_loss = train_epoch_on_indices(
            probe,
            opt,
            X_train,
            y_train,
            perm,
            batch_size,
            desc=f'Final batches (L{layer:02d}) ep {epoch}/{train_epochs}',
            position=2,
        )
        epoch_iter.set_postfix(
            avg_loss=float(avg_loss),
            best=float(best_loss),
            since_best=since_best,
        )

        if avg_loss < best_loss - 1e-9:
            best_loss = avg_loss
            best_state = {
                key: val.detach().cpu().clone()
                for key, val in probe.state_dict().items()
            }
            since_best = 0
        else:
            since_best += 1

        if early_gap > 0 and since_best >= early_gap:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)
        probe.to(X_train.device)

    final_drift = float(compute_probe_drift(probe, init_state)['total'])
    return probe, final_drift, float(best_loss)


@hydra.main(version_base=None, config_path='.', config_name='config_init_study')
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.INFO)
    log.info('Loaded config:\n' + OmegaConf.to_yaml(cfg))
    validate_cfg(cfg)

    seed = int(cfg.mdl.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    train_feats, train_labels_np, train_meta, train_meta_raw = load_cache_dir(
        cfg.dataset.train
    )
    test_feats, test_labels_np, test_meta, test_meta_raw = load_cache_dir(
        cfg.dataset.test
    )
    if (train_meta.hidden_size != test_meta.hidden_size or
            train_meta.num_states != test_meta.num_states):
        raise ValueError(
            f'Train/Test mismatch: train(L={train_meta.num_states},'
            f'D={train_meta.hidden_size}) '
            f'test(L={test_meta.num_states},D={test_meta.hidden_size})'
        )

    n_train, layer_count, hidden_dim = train_meta.shape
    n_test = test_meta.shape[0]
    log.info(
        f'Train cached feats: N={n_train} L={layer_count} D={hidden_dim}, dtype={train_meta.dtype}'
    )
    log.info(
        f'Test cached feats: N={n_test} L={layer_count} D={hidden_dim}, dtype={test_meta.dtype}'
    )

    split_num = int(cfg.mdl.split_num)
    split_ratio = float(cfg.mdl.split_ratio)
    splits = geometric_split_indices(
        n_train,
        split_num=split_num,
        ratio=split_ratio,
        seed=seed,
    )
    log.info(f'Geometric splits sizes: {[len(split) for split in splits]}')

    y_train_all = torch.from_numpy(train_labels_np).to(device)
    y_test_all = torch.from_numpy(test_labels_np).to(device)
    probe_width = int(list(cfg.mdl.probe_hidden_size)[0])
    init_strategy = str(cfg.mdl.init_strategy)
    init_gamma = cfg.mdl.get('init_gamma', None)
    if init_gamma is not None:
        init_gamma = float(init_gamma)

    mdl_sum_by_layer: Dict[int, float] = {}
    nll_by_layer: Dict[int, float] = {}
    stop_epoch_by_layer: Dict[int, List[int]] = {}
    test_acc_by_layer: Dict[int, float] = {}
    test_loss_by_layer: Dict[int, float] = {}
    drift_final_by_layer: Dict[int, float] = {}

    effective_lr = get_effective_lr(
        base_lr=float(cfg.mdl.lr),
        strategy=init_strategy,
        width=probe_width
    )
    lr_rule = get_lr_rule(init_strategy)
    forward_prefactor = get_forward_prefactor(init_strategy, probe_width)
    bias_rule = get_bias_rule(init_strategy)

    for layer in tqdm(range(layer_count), desc='Layers', position=0, leave=True):
        X_train = torch.from_numpy(
            np.asarray(train_feats[:, layer, :], dtype=np.float32).copy()
        ).to(device)
        X_test = torch.from_numpy(
            np.asarray(test_feats[:, layer, :], dtype=np.float32).copy()
        ).to(device)

        log.info(f'\n[Layer {layer:02d}] Online MDL...')
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
            f'[Layer {layer:02d}] total_MDL(sum CE)={mdl_sum:.2f}, '
            f'encoded={n_encoded}, NLL={nll_by_layer[layer]:.4f}'
        )

        log.info(f'[Layer {layer:02d}] Final train + test eval...')
        final_probe, final_drift, best_loss = train_final_probe_full_train_tensor(
            X_train,
            y_train_all,
            cfg,
            layer=layer,
        )
        test_idx = torch.arange(n_test, device=device, dtype=torch.long)
        test_metrics = eval_acc_loss_on_indices(
            final_probe,
            X_test,
            y_test_all,
            test_idx,
            batch_size=int(cfg.mdl.batch_size),
            desc=f'Test eval (layer {layer:02d})',
            position=1,
        )
        test_acc_by_layer[layer] = float(test_metrics['acc'])
        test_loss_by_layer[layer] = float(test_metrics['loss'])
        drift_final_by_layer[layer] = final_drift

        log.info(
            f'[Layer {layer:02d}] test_acc={test_acc_by_layer[layer]:.4f} '
            f'test_loss={test_loss_by_layer[layer]:.4f} '
            f'drift={drift_final_by_layer[layer]:.4f}'
        )

        del X_train, X_test, final_probe
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)
    results = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'train_meta': train_meta_raw,
        'test_meta': test_meta_raw,
        'splits': [int(len(split)) for split in splits],
        'mdl_sum_ce': mdl_sum_by_layer,
        'mdl_nll': nll_by_layer,
        'stopping_epochs_per_stage': stop_epoch_by_layer,
        'test_acc': test_acc_by_layer,
        'test_loss': test_loss_by_layer,
        'drift_final_total_l2': drift_final_by_layer,
        'probe_type': str(cfg.mdl.probe_type),
        'probe_hidden_size': list(cfg.mdl.probe_hidden_size),
        'init_strategy': init_strategy,
        'init_gamma': init_gamma,
        'effective_lr': effective_lr,
        'lr_scaling_rule': lr_rule,
        'bias_init': bias_rule,
        'forward_prefactor': forward_prefactor,
        'probe_width': probe_width,
        'L': int(layer_count),
        'D': int(hidden_dim),
    }

    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)

    best_layer = min(mdl_sum_by_layer.keys(), key=lambda key: mdl_sum_by_layer[key])
    log.info(
        f'\n[BEST by MDL] layer={best_layer} MDL={mdl_sum_by_layer[best_layer]:.2f} '
        f'NLL={nll_by_layer[best_layer]:.4f}'
    )
    log.info(f'[Outputs] {out_dir}')


if __name__ == '__main__':
    OmegaConf.register_new_resolver('calc_path', get_custom_dir)
    main()
