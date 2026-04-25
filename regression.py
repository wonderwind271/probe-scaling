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
    emb = np.concatenate([pos_cls, neg_cls], axis=0)
    return emb.mean(), emb.var()


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
    try:
        with open(f'{dataset}_representation_stats.pkl', 'rb') as f:
            representation_stats = pickle.load(f)
    except:
        representation_stats = {}
        for pos, neg in path_ls:
            if dataset == 'mscoco':
                model, task, _ = pos.split('-')
            elif dataset == 'openimages':
                model, task, _ = pos.split('/')
            for layer in range(layers):
                print(f'Loading representation for {model} {task} layer {layer}...')
                mean, var = load_representation(emb_path / pos, emb_path / neg, layer, model)
                representation_stats[(model, task, layer)] = (mean, var)

        with open(f'{dataset}_representation_stats.pkl', 'wb') as f:
            pickle.dump(representation_stats, f)

    X, label = [], []
    # for mscoco, X = 480: 12 layers, 4 tasks, 5 seeds, 2 models
    for (model, task, *_, layer), width in min_mdl.items():
        mean, var = representation_stats[(model, task, layer)]
        X.append([mean, var, layer, model_dim[model]])
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
    print(reg.summary(xname=['mean', 'var', 'layer', 'model_d']))


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
