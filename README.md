# probe-scaling
Probe model's scaling behavior and baseline

## File structure
- `probe_model.py`: Given the model, define classifier model on the hidden layers
- `train_probe.py`: Given the model, train the classifier model on the hidden layers (use UD dataset)
- `train.py`: Use adversary method to train the model so that it can't tell two types of the sentences apart

## Huggingface Dataset and Models
- [Seed42Lab/en-ud-train-pair](https://huggingface.co/datasets/Seed42Lab/en-ud-train-pair): dataset for adversary training (can also be used for classifier training)
- [Seed42Lab/en-ud-test](https://huggingface.co/datasets/Seed42Lab/en-ud-test): test set for classifier accuracy
