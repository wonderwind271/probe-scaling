'''Do linear regression on the MSCoco dataset.
Predict the best width (log-scaled) according to the input features and model information: (1) mean and var of the representations, (2) the number of layers, (3) size of model's representation.

Results: probe-scaling/outputs/
Hidden representations: Under vlm-lens/'''
from collections import defaultdict
import json
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from pathlib import Path
import numpy as np
import pickle
import torch
from sklearn.decomposition import PCA


def load_representation(pos_dir: Path, neg_dir: Path, layer_idx: int, model: str):
    pos_npy = pos_dir / f'layer_{layer_idx:02d}_cls.npy'
    neg_npy = neg_dir / f'layer_{layer_idx:02d}_cls.npy'
    if pos_npy.exists() and neg_npy.exists():
        pos_cls = np.load(pos_npy)
        neg_cls = np.load(neg_npy)
    else:
        pos_cls = torch.load(pos_dir / f'layer_{layer_idx:02d}.pt', map_location='cpu').squeeze()
        neg_cls = torch.load(neg_dir / f'layer_{layer_idx:02d}.pt', map_location='cpu').squeeze()
        if pos_cls.ndim == 3 and pos_cls.size(-1) == model_dim[model]:  # (batch_size, seq_len, hidden_size)
            pos_cls = pos_cls[:, 0, :].float().numpy()
            neg_cls = neg_cls[:, 0, :].float().numpy()
    return pos_cls, neg_cls


def stats_representation(pos_cls, neg_cls):
    emb = np.concatenate([pos_cls, neg_cls], axis=0)
    return emb.mean(), emb.var()


def pca_representation(pos_cls: np.ndarray, neg_cls: np.ndarray, _n_components: int = 20):
    emb = np.concatenate([pos_cls, neg_cls], axis=0)
    pca = PCA(n_components=_n_components, svd_solver='randomized', random_state=42)
    pca.fit(emb)
    return float(np.sum(pca.explained_variance_ratio_))


def load_test_acc(result_dir) -> dict[tuple[str, str, int], float]:
    """Average linear-probe test accuracy per (model, task, layer) across seeds.

    Reads results where probe_type == 'linear' (single (d_in, d_out) layer, no hidden size). Dedup key is (model, task, seed); latest timestamp wins.

    Returns: dict mapping (model, task, layer) -> float
    """
    acc_records = defaultdict(list)
    seen = {}
    for result_file in result_dir.rglob('results.json'):
        with open(result_file) as f:
            result = json.load(f)
        if result['config']['mdl']['probe_type'] != 'linear':
            continue
        model = result['config']['model_short']
        task = result['config']['task_name']
        seed = int(result['config']['mdl']['seed'])
        timestamp = result_file.parts[-2]
        key = (model, task, seed)
        if key in seen and timestamp <= seen[key]:
            continue
        seen[key] = timestamp
        for layer, acc in result['test_acc'].items():
            acc_records[(model, task, int(layer))].append(acc)
    return {k: float(np.mean(v)) for k, v in acc_records.items()}


def load_pca_variance(path_ls, emb_path, layers, dataset, n_components=10):
    """Variance explained by the top-n_components PCs per (model, task, layer).
    Uses .npy CLS-token cache when available; falls back to .pt files.
    Returns: dict mapping (model, task, layer) -> float in [0, 1]
    """
    pca_var = {}
    for pos, neg in path_ls:
        if dataset == 'mscoco':
            model, task, _ = pos.split('-')
        elif dataset == 'openimages':
            model, task, _ = pos.split('/')
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

        pos_dir = emb_path / pos
        neg_dir = emb_path / neg

        for layer in range(layers):
            pos_npy = pos_dir / f'layer_{layer:02d}_cls.npy'
            neg_npy = neg_dir / f'layer_{layer:02d}_cls.npy'
            if pos_npy.exists() and neg_npy.exists():
                pos_cls = np.load(pos_npy)
                neg_cls = np.load(neg_npy)
            else:
                pos_t = torch.load(pos_dir / f'layer_{layer:02d}.pt', map_location='cpu')
                neg_t = torch.load(neg_dir / f'layer_{layer:02d}.pt', map_location='cpu')
                pos_cls = pos_t[:, 0, :].float().numpy()
                neg_cls = neg_t[:, 0, :].float().numpy()
                del pos_t, neg_t

            X = np.concatenate([pos_cls, neg_cls], axis=0).astype(np.float32)
            pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
            pca.fit(X)
            pca_var[(model, task, layer)] = float(np.sum(pca.explained_variance_ratio_))

    return pca_var


def explore(X: np.ndarray, label: np.ndarray, feature_names: list) -> None:
    """Exploratory diagnostics: correlation matrix, partial correlations,
    and pairwise residual analysis after partialling out each feature.
    """
    from sklearn.linear_model import LinearRegression as LR

    X = np.array(X, dtype=float)
    label = np.array(label, dtype=float)
    n = len(feature_names)

    # 1. Full correlation matrix (features + label)
    data = np.column_stack([X, label])
    names = feature_names + ['label']
    corr = np.corrcoef(data.T)
    print('\n=== Correlation matrix ===')
    print(f'{"":>14}' + ''.join(f'{nm:>14}' for nm in names))
    for i, nm in enumerate(names):
        print(f'{nm:>14}' + ''.join(f'{corr[i, j]:>14.3f}' for j in range(len(names))))

    # 2. Partial correlations: feature j vs label, controlling for all other features
    print('\n=== Partial correlations with label (others partialled out) ===')
    for j, name in enumerate(feature_names):
        others = [k for k in range(n) if k != j]
        X_others = X[:, others]
        feat_resid = X[:, j] - LR().fit(X_others, X[:, j]).predict(X_others)
        label_resid = label - LR().fit(X_others, label).predict(X_others)
        pc = np.corrcoef(feat_resid, label_resid)[0, 1]
        print(f'  partial_corr({name:>12}, label | rest) = {pc:+.4f}')

    # 3. For each pair (feat, control), partial out control from feat and from label
    print('\n=== Residual correlations after partialling out one control ===')
    for ctrl_name in feature_names:
        ctrl_idx = feature_names.index(ctrl_name)
        ctrl_col = X[:, ctrl_idx].reshape(-1, 1)
        label_resid = label - LR().fit(ctrl_col, label).predict(ctrl_col)
        for j, name in enumerate(feature_names):
            if j == ctrl_idx:
                continue
            feat_resid = X[:, j] - LR().fit(ctrl_col, X[:, j]).predict(ctrl_col)
            r = np.corrcoef(feat_resid, label_resid)[0, 1]
            print(f'  corr({name:>12} | {ctrl_name}, label | {ctrl_name}) = {r:+.4f}')


def load_results(result_dir, avg_seed=False):
    # load all `results.json` files under result_dir
    results = defaultdict(lambda: defaultdict(dict))
    seen = {}  # (model, task, hidden_size, seed) -> latest timestamp string
    for result_file in result_dir.rglob('results.json'):
        with open(result_file, 'r') as f:
            result = json.load(f)
            if result['config']['mdl']['probe_type'] != 'mlp':
                continue
            model = result['config']['model_short']
            task = result['config']['task_name']
            hidden_size = int(result['config']['mdl']['probe_hidden_size'][0])
            seed = int(result['config']['mdl']['seed'])
            timestamp = result_file.parts[-2]  # e.g. '2026-03-22_21-54-46'
            key = (model, task, hidden_size, seed)
            if key in seen:
                if timestamp <= seen[key]:
                    print(f'[warn] skipping older duplicate: {result_file}')
                    continue
                print(f'[warn] replacing older run for {key}')
            seen[key] = timestamp
            for layer, mdl in result['mdl_sum_ce'].items():
                results[(model, task, seed)][int(layer)][hidden_size] = mdl

    min_mdl = {}
    if avg_seed:
        tmp = defaultdict(lambda: defaultdict(dict))
        for (model, task, seed), layer_dict in results.items():
            for layer, mdl_dict in layer_dict.items():
                for size, mdl in mdl_dict.items():
                    tmp[(model, task, layer)][size][seed] = mdl
        for key, layer_dict in tmp.items():
            tmp_ = {size: np.mean(list(layer_dict[size].values())) for size in layer_dict}
            min_mdl[key] = min(tmp_, key=tmp_.get)

    else:
        for (model, task, seed), layer_dict in results.items():
            for layer, mdl_dict in layer_dict.items():
                min_mdl[(model, task, seed, layer)] = min(mdl_dict, key=mdl_dict.get)
    return min_mdl


def regression(path_ls, emb_path, results_path, dataset='mscoco', layers=12, log_width=False, avg_seed=False):
    # find the best width
    min_mdl = load_results(results_path, avg_seed=avg_seed)
    linear_probe = load_test_acc(results_path)
    try:
        with open(f'{dataset}_representation_stats.pkl', 'rb') as f:
            representation_stats = pickle.load(f)
    except (FileNotFoundError, Exception):
        representation_stats = {}
        for pos, neg in path_ls:
            if dataset == 'mscoco':
                model, task, _ = pos.split('-')
            elif dataset == 'openimages':
                model, task, _ = pos.split('/')
            for layer in range(layers):
                print(f'Loading representation for {model} {task} layer {layer}...')
                pos_cls, neg_cls = load_representation(emb_path / pos, emb_path / neg, layer, model)
                mean, var = stats_representation(pos_cls, neg_cls)
                print(f'Finish loading representation for {model} {task} layer {layer}')
                pca = pca_representation(pos_cls, neg_cls, _n_components=20)
                linear_acc = linear_probe[(model, task, layer)]
                representation_stats[(model, task, layer)] = (mean, var, pca, linear_acc)

        with open(f'{dataset}_representation_stats.pkl', 'wb') as f:
            pickle.dump(representation_stats, f)

    X, label = [], []
    # for mscoco, X = 480: 12 layers, 4 tasks, 5 seeds, 2 models
    for (model, task, *_, layer), width in min_mdl.items():
        mean, var, pca, linear_acc = representation_stats[(model, task, layer)]
        X.append([pca, linear_acc, layer])
        # X.append([layer])
        if log_width:
            width = np.log(width)
        label.append(width)
    reg = LinearRegression().fit(X, label)
    print('R2: ', reg.score(X, label))
    print('Coefficients: ', reg.coef_)
    print('Intercept: ', reg.intercept_)

    # X: (intercept, mean, var, layer_idx, model_d)
    X_const = sm.add_constant(np.array(X, dtype=float))
    reg = sm.OLS(label, X_const, hasconst=True).fit()
    print(reg.params)
    print(reg.summary(xname=['const', 'pca', 'linear_acc', 'layer']))
    # print(reg.summary(xname=['const', 'layer']))

    explore(np.array(X), np.array(label), ['pca', 'linear_acc', 'layer'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mscoco', choices=['mscoco', 'openimages'], help='Dataset to use for regression.')
    parser.add_argument('--log_width', action='store_true', help='Whether to log-transform the width before regression.')
    parser.add_argument('--avg_seed', action='store_true', help='Whether to average results across seeds.')
    args = parser.parse_args()

    # mscoco
    if args.dataset == 'mscoco':
        path_ls = [(f'{model}-{obj}-pos', f'{model}-{obj}-neg') for obj in ['car', 'chair', 'person', 'table'] for model in ['clip', 'dino']]
        emb_path = Path('/home/xxluo/projects/aip-fredashi/xxluo/myproject/vlm-lens')
        results_path = Path('outputs/mscoco/')

    # openimages
    if args.dataset == 'openimages':
        path_ls = [(f'{model}/{obj}/positive', f'{model}/{obj}/negative')
                   for obj in ['bicycle', 'car', 'dog', 'door', 'flower', 'house', 'train', 'table', 'tree', 'window'] for model in ['clip', 'dino']]
        emb_path = Path('/home/xxluo/projects/aip-fredashi/xxluo/myproject/openimages-tokens')
        results_path = Path('outputs/openimage/')

    model_dim = {'clip': 768, 'dino': 768}
    regression(path_ls, emb_path, results_path, dataset=args.dataset, log_width=args.log_width, avg_seed=args.avg_seed)

    # regression(path_ls, emb_path, results_path, dataset=args.dataset, log_width=True, avg_seed=True)
