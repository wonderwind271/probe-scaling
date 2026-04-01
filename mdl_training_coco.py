'''Train binary probes on cached COCO hidden states and report MDL.'''

from __future__ import annotations
import json
import logging
import os
from typing import Dict, List, Tuple
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from mdl_training_cached import (
    build_probe,
    geometric_split_indices,
    get_custom_dir,
    make_optimizer,
)

log = logging.getLogger(__name__)


def load_layer_datasets(
    positive_dir: str,
    negative_dir: str,
    layer_idx: int,
    test_ratio: float,
    flat_patch: bool = False,  # True: take [patch_num x D] features for probing. False: take [CLS]
    seed: int = 42,
) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset, torch.Size]:
    '''Load one layer of cached positive and negative tensors and split train/test.'''
    positive_path = os.path.join(positive_dir, f'layer_{layer_idx:02d}.pt')
    negative_path = os.path.join(negative_dir, f'layer_{layer_idx:02d}.pt')
    # positive_tensor = torch.load(positive_path, map_location='cpu').squeeze()
    # negative_tensor = torch.load(negative_path, map_location='cpu').squeeze()
    positive_tensor = torch.load(positive_path, map_location='cpu')
    negative_tensor = torch.load(negative_path, map_location='cpu')
    
    if flat_patch:
        positive_tensor = positive_tensor.reshape(positive_tensor.size(0), -1)
        negative_tensor = negative_tensor.reshape(negative_tensor.size(0), -1)
    else:
        positive_tensor = positive_tensor[:, 0, :]
        negative_tensor = negative_tensor[:, 0, :]

    feature_shape = positive_tensor.shape
    num_positive = feature_shape[0]
    num_negative = negative_tensor.shape[0]
    num_test = round((num_positive + num_negative) * test_ratio)

    features = torch.cat((positive_tensor, negative_tensor), dim=0).float()
    labels = torch.cat((torch.ones(num_positive, dtype=torch.int64), 
                        torch.zeros(num_negative, dtype=torch.int64)), dim=0)

    full_dataset = torch.utils.data.TensorDataset(features, labels)
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - num_test, num_test],
        generator=split_generator,
    )
    return train_dataset, test_dataset, feature_shape


def make_subset_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
):
    '''Build a dataloader for a dataset or subset.'''
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


@torch.no_grad()
def compute_dataset_ce_sum(
    model: nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> float:
    '''Compute the summed cross-entropy on a dataset for MDL.'''
    model.eval()
    total_loss = 0.0
    data_loader = make_subset_loader(dataset, batch_size=batch_size, shuffle=False)
    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        total_loss += float(F.cross_entropy(logits, labels, reduction='sum').item())
    return total_loss


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    '''Evaluate probe accuracy and mean cross-entropy on a dataset.'''
    model.eval()
    correct = 0
    loss_sum = 0.0
    data_loader = make_subset_loader(dataset, batch_size=batch_size, shuffle=False)

    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss_sum += float(F.cross_entropy(logits, labels, reduction='sum').item())
        predictions = logits.argmax(dim=-1)
        correct += int((predictions == labels).sum().item())

    total = len(dataset)
    return {
        'acc': correct / max(total, 1),
        'loss': loss_sum / max(total, 1),
        'n': float(total),
    }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset,
    batch_size: int,
    device: torch.device,
) -> float:
    '''Train a probe for one epoch and return the average training loss.'''
    model.train()
    loss_sum = 0.0
    data_loader = make_subset_loader(dataset, batch_size=batch_size, shuffle=True)

    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * int(labels.size(0))

    return loss_sum / max(len(dataset), 1)


def train_mdl_stage(
    probe: nn.Module,
    train_subset,
    encode_subset,
    batch_size: int,
    train_epochs: int,
    lr: float,
    optimizer_name: str,
    early_gap: int,
    mdl_prefix: float,
    layer_idx: int,
    stage_idx: int,
    num_stages: int,
    device: torch.device,
) -> Tuple[float, int]:
    '''Train one online-MDL stage with early stopping on encoding loss.'''
    probe.to(device)
    optimizer = make_optimizer(optimizer_name, probe.parameters(), lr)

    best_total_mdl = float('inf')
    best_encode_ce = float('inf')
    best_epoch = 1
    epochs_since_best = 0

    for epoch in range(1, train_epochs + 1):
        _ = train_one_epoch(
            probe,
            optimizer,
            train_subset,
            batch_size=batch_size,
            device=device,
        )

        encode_ce_sum = compute_dataset_ce_sum(
            probe,
            encode_subset,
            batch_size=batch_size,
            device=device,
        )
        total_mdl = mdl_prefix + encode_ce_sum

        if total_mdl < best_total_mdl - 1e-9:
            best_total_mdl = total_mdl
            best_encode_ce = encode_ce_sum
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        log.info(
            f'[Layer {layer_idx:02d} Stage {stage_idx}/{num_stages}] '
            f'epoch={epoch} encode_ce_sum={encode_ce_sum:.2f} '
            f'total_mdl={total_mdl:.2f} best_total_mdl={best_total_mdl:.2f} '
            f'since_best={epochs_since_best}',
        )

        if early_gap > 0 and epochs_since_best >= early_gap:
            break

    return best_encode_ce, best_epoch


def run_online_mdl(
    train_dataset: torch.utils.data.TensorDataset,
    stage_splits: List[np.ndarray],
    cfg: DictConfig,
    input_dim: int,
    layer_idx: int,
    device: torch.device,
) -> Tuple[float, int, List[int]]:
    '''Run online MDL over geometric train-set stages for one layer.'''
    batch_size = int(cfg.mdl.batch_size)
    probe_type = str(cfg.mdl.probe_type)
    hidden_sizes = (
        list(cfg.mdl.probe_hidden_size)
        if probe_type.lower() != 'linear'
        else []
    )
    train_epochs = int(cfg.mdl.train_epochs)
    early_gap = int(cfg.mdl.early_stopping_gap)
    lr = float(cfg.mdl.lr)
    optimizer_name = str(cfg.mdl.optimizer)

    mdl_sum = 0.0
    total_encoded = 0
    best_epochs: List[int] = []
    num_stages = len(stage_splits) - 1
    train_indices = stage_splits[0]

    stage_bar = tqdm(
        range(num_stages),
        desc=f'MDL stages (layer {layer_idx:02d})',
        position=1, leave=False,
    )
    for stage_offset in stage_bar:
        stage_idx = stage_offset + 1
        encode_indices = np.sort(stage_splits[stage_offset + 1])

        train_subset = torch.utils.data.Subset(
            train_dataset,
            train_indices.astype(np.int64).tolist(),
        )
        encode_subset = torch.utils.data.Subset(
            train_dataset,
            encode_indices.astype(np.int64).tolist(),
        )

        probe, _ = build_probe(
            probe_type,
            d_in=input_dim,
            hidden_sizes=hidden_sizes,
            num_classes=2,
        )

        best_encode_ce, best_epoch = train_mdl_stage(
            probe=probe,
            train_subset=train_subset,
            encode_subset=encode_subset,
            batch_size=batch_size,
            train_epochs=train_epochs,
            lr=lr,
            optimizer_name=optimizer_name,
            early_gap=early_gap,
            mdl_prefix=mdl_sum,
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            num_stages=num_stages,
            device=device,
        )
        best_epochs.append(best_epoch)

        mdl_sum += best_encode_ce
        total_encoded += len(encode_indices)
        train_indices = np.concatenate(
            [train_indices, encode_indices],
            axis=0,
        )

        stage_bar.set_postfix(mdl=float(mdl_sum), encoded=total_encoded, best_ep=best_epoch)

    return mdl_sum, total_encoded, best_epochs


def train_final_probe(
    train_dataset,
    cfg: DictConfig,
    *,
    layer_idx: int,
    input_dim: int,
    device: torch.device,
) -> nn.Module:
    '''Train a final probe on the full training set for one layer.'''
    batch_size = int(cfg.mdl.batch_size)
    probe_type = str(cfg.mdl.probe_type)
    hidden_sizes = (
        list(cfg.mdl.probe_hidden_size)
        if probe_type.lower() != 'linear'
        else []
    )
    train_epochs = int(cfg.mdl.train_epochs)
    early_gap = int(cfg.mdl.early_stopping_gap)
    lr = float(cfg.mdl.lr)
    optimizer_name = str(cfg.mdl.optimizer)

    probe, _ = build_probe(
        probe_type,
        d_in=input_dim,
        hidden_sizes=hidden_sizes,
        num_classes=2,
    )
    probe.to(device)
    optimizer = make_optimizer(optimizer_name, probe.parameters(), lr)

    best_loss = float('inf')
    best_state = None
    epochs_since_best = 0

    epoch_bar = tqdm(
        range(1, train_epochs + 1),
        desc=f'Final train (layer {layer_idx:02d})',
        position=1,
        leave=False,
    )
    for _epoch in epoch_bar:
        avg_loss = train_one_epoch(
            probe,
            optimizer,
            train_dataset,
            batch_size=batch_size,
            device=device,
        )
        epoch_bar.set_postfix(
            avg_loss=float(avg_loss),
            best=float(best_loss),
            since_best=epochs_since_best,
        )

        if avg_loss < best_loss - 1e-9:
            best_loss = avg_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if early_gap > 0 and epochs_since_best >= early_gap:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)
        probe.to(device)
    return probe


@hydra.main(version_base=None, config_path='.', config_name='config_cached_coco')
def main(cfg: DictConfig):
    '''Run MDL probe training for every cached COCO layer.'''
    logging.getLogger().setLevel(logging.INFO)
    log.info('Loaded config:\n' + OmegaConf.to_yaml(cfg))

    seed = int(cfg.mdl.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    num_stage_splits = 5
    split_ratio = 2.0
    mdl_sum_by_layer: Dict[int, float] = {}
    nll_by_layer: Dict[int, float] = {}
    stop_epochs_by_layer: Dict[int, List[int]] = {}
    test_acc_by_layer: Dict[int, float] = {}
    test_loss_by_layer: Dict[int, float] = {}
    split_sizes = None
    feature_dim = None

    for layer_idx in tqdm(range(cfg.layer), desc='Layers', position=0, leave=True):
        train_dataset, test_dataset, feature_shape = load_layer_datasets(
            positive_dir=cfg.dataset.pos,
            negative_dir=cfg.dataset.neg,
            layer_idx=layer_idx,
            test_ratio=cfg.dataset.test_portion,
            seed=seed,
        )

        input_dim = feature_shape[-1]
        num_train = len(train_dataset)
        num_test = len(test_dataset)
        log.info(
            f'Train/Test cached feature: N_train={num_train}/N_test={num_test} D={input_dim}',
        )

        stage_splits = geometric_split_indices(
            num_train,
            split_num=num_stage_splits,
            ratio=split_ratio,
            seed=seed,
        )
        log.info(f'Geometric splits sizes: {[len(split) for split in stage_splits]}')

        log.info(f'\n[Layer {layer_idx:02d}] Online MDL...')
        mdl_sum, n_encoded, best_epochs = run_online_mdl(
            train_dataset=train_dataset,
            stage_splits=stage_splits,
            cfg=cfg,
            input_dim=int(input_dim),
            layer_idx=layer_idx,
            device=device,
        )
        mdl_sum_by_layer[layer_idx] = float(mdl_sum)
        nll_by_layer[layer_idx] = float(mdl_sum) / max(int(n_encoded), 1)
        stop_epochs_by_layer[layer_idx] = best_epochs

        log.info(
            f'[Layer {layer_idx:02d}] total_MDL(sum CE)={mdl_sum:.2f}, '
            f'encoded={n_encoded}, NLL={nll_by_layer[layer_idx]:.4f}',
        )
        log.info(
            f'[Layer {layer_idx:02d}] stopping epochs per stage = {best_epochs}',
        )

        log.info(f'[Layer {layer_idx:02d}] Final train + test eval...')
        final_probe = train_final_probe(
            train_dataset,
            cfg,
            layer_idx=layer_idx,
            input_dim=int(input_dim),
            device=device,
        )
        test_metrics = evaluate_probe(
            final_probe,
            test_dataset,
            batch_size=int(cfg.mdl.batch_size),
            device=device,
        )
        test_acc_by_layer[layer_idx] = float(test_metrics['acc'])
        test_loss_by_layer[layer_idx] = float(test_metrics['loss'])
        log.info(
            f'[Layer {layer_idx:02d}] test_acc={test_acc_by_layer[layer_idx]:.4f} '
            f'test_loss={test_loss_by_layer[layer_idx]:.4f}',
        )

        del train_dataset, test_dataset, final_probe
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if split_sizes is None:
            split_sizes = [int(len(split)) for split in stage_splits]
        if feature_dim is None:
            feature_dim = int(input_dim)

    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)

    results = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'splits': split_sizes,
        'mdl_sum_ce': mdl_sum_by_layer,
        'mdl_nll': nll_by_layer,
        'stopping_epochs_per_stage': stop_epochs_by_layer,
        'test_acc': test_acc_by_layer,
        'test_loss': test_loss_by_layer,
        'probe_type': str(cfg.mdl.probe_type),
        'probe_hidden_size': (
            list(cfg.mdl.probe_hidden_size)
            if str(cfg.mdl.probe_type).lower() != 'linear'
            else []
        ),
        'D': feature_dim,
    }

    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)

    best_layer = min(mdl_sum_by_layer.keys(), key=lambda key: mdl_sum_by_layer[key])
    log.info(
        f'\n[BEST by MDL] layer={best_layer} '
        f'MDL={mdl_sum_by_layer[best_layer]:.2f} '
        f'NLL={nll_by_layer[best_layer]:.4f}',
    )
    log.info(f'[Outputs] {out_dir}')


if __name__ == '__main__':
    OmegaConf.register_new_resolver('calc_path', get_custom_dir)
    main()
