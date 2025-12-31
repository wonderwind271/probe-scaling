# probe-scaling
Probe model's scaling behavior and baseline

## File structure
- `probe_model.py`: Given the model, define classifier model on the hidden layers
- `train_probe.py`: Given the model, train the classifier model on the hidden layers (use UD dataset)
- `train.py`: Use adversary method to train the model so that it can't tell two types of the sentences apart

## Huggingface Dataset and Models
- [Seed42Lab/en-ud-train-pair](https://huggingface.co/datasets/Seed42Lab/en-ud-train-pair): dataset for adversary training (can also be used for classifier training)
- [Seed42Lab/en-ud-test](https://huggingface.co/datasets/Seed42Lab/en-ud-test): test set for classifier accuracy

## Formula
### Fine-tuning Objective

In order to make the model **unable** to tell $x^{+}$ and $x^{-}$ apart at layer $L$, we use

$$
\mathcal{L} =
\lambda_{\mathrm{conf}}
\sum_{k = k_0}^{L}
w_k
\,
\bigl\|
h^{k}(x^{+}) - h^{k}(x^{-})
\bigr\|_2^{2}
\;+\;
\lambda_{\mathrm{KL}}
\,
\mathrm{KL}\!\left(
p_{\theta}(\cdot \mid x)
\,\middle\|\,
p_{\theta_0}(\cdot \mid x)
\right)
\;+\;
\lambda_{\Delta}
\,
\|\Delta\|_2^{2}
$$

As the training objective. $k_0\leq L$ but not necessarily equal, because otherwise we are forcing the model to abruptly unlearn a feature at a certain layer. We also apply KL penalty and regularization term $\lambda_{\Delta}\,\|\Delta\|_2^{2}$ ($\Delta$ is the added part of the model, similar to LoRA but we do not use low rank approximation).
