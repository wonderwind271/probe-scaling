import math
from typing import List
from datasets import Dataset, load_dataset


def geometric_splits(N: int, split_num: int, ratio: float) -> List[int]:
    assert N > split_num, "N must be greater than split_num"
    assert ratio > 1.0, "ratio must be > 1"
    assert split_num >= 1

    weights = [ratio ** i for i in range(split_num)]
    weight_sum = sum(weights)

    raw_sizes = [w / weight_sum * N for w in weights]

    sizes = [math.floor(x) for x in raw_sizes[:-1]]
    sizes.append(N - sum(sizes))  # last split absorbs remainder

    assert sum(sizes) == N
    return sizes


def split_dataset(
    dataset: Dataset,
    split_num: int,
    ratio: float,
    seed: int,
) -> List[Dataset]:
    """
    Split a HuggingFace Dataset into geometric chunks for online MDL.

    Args:
        dataset: HuggingFace Dataset
        split_num: number of splits
        ratio: geometric ratio (>1)
        seed: shuffle seed; if seed < 0, do not shuffle

    Returns:
        List[Dataset]: [D1, D2, ..., D_split_num]
    """
    N = len(dataset)
    assert N > split_num, "Dataset too small for requested splits"

    if seed >= 0:
        dataset = dataset.shuffle(seed=seed)

    sizes = geometric_splits(N, split_num, ratio)

    splits = []
    start = 0
    for size in sizes:
        end = start + size
        splits.append(dataset.select(range(start, end)))
        start = end

    return splits
