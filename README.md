# probe-scaling
Probe model's scaling behavior and baseline

## File structure
- `cache_hidden_states.py`: Cache last-token (or predicate+last) hidden states to disk for cheap probing
- `mdl_training_cached.py`: Train/eval probes from cached hidden states (no model inference)
- `probe_model.py`: Probe architectures used by `mdl_training_cached.py`
- Archived scripts/configs: `archive/` (e.g. `archive/train.py`, `archive/train_probe.py`, `archive/mdl_training.py`)

## Huggingface Dataset and Models
- [Seed42Lab/en-ud-train-pair](https://huggingface.co/datasets/Seed42Lab/en-ud-train-pair): dataset for adversary training (can also be used for classifier training)
- [Seed42Lab/en-ud-test](https://huggingface.co/datasets/Seed42Lab/en-ud-test): test set for classifier accuracy

## Probing workflow
This repo supports a two-stage workflow:
1) **Cache hidden states**: `cache_hidden_states.py` (+ `config_cache_hidden.yaml`)
2) **Train probes from caches**: `mdl_training_cached.py` (+ `config_cached.yaml`)

## Formula
### Fine-tuning Objective

In order to make the model **unable** to tell $x^{+}$ and $x^{-}$ apart at layer $L$, we use

$$
\mathcal{L} =
\lambda_{\mathrm{conf}}
\sum_{k = k_0}^{L}
w_k
\bigl\|
h^{k}(x^{+}) - h^{k}(x^{-})
\bigr\|_2^{2}
+
\lambda_{\mathrm{KL}}
\mathrm{KL}\left(
p_{\theta}(\cdot \mid x)
||
p_{\theta_0}(\cdot \mid x)
\right)
+
\lambda_{\Delta}
\|\Delta\|_2^{2}
$$

As the training objective. $k_0\leq L$ but not necessarily equal, because otherwise we are forcing the model to abruptly unlearn a feature at a certain layer. We also apply KL penalty and regularization term $\lambda_{\Delta}\,\|\Delta\|_2^{2}$ ($\Delta$ is the added part of the model, similar to LoRA but we do not use low rank approximation).
