import torch
from datasets import load_dataset, DatasetDict, ClassLabel, Value

DATASET_REGISTRY = {}
COLLATE_FN_REGISTRY = {}

def register_dataset(name):
    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator

def register_collate_fn(name):
    def decorator(fn):
        COLLATE_FN_REGISTRY[name] = fn
        return fn
    return decorator

def get_collate_fn(name):
    return COLLATE_FN_REGISTRY.get(name)

@register_dataset("sudoku")
def load_sudoku(cache_dir, filter_train : bool = False) -> DatasetDict:
    """
    Preprocesses the Sudoku (https://huggingface.co/datasets/sapientinc/sudoku-extreme) dataset.

    The dataset is used in the Hierarchical Reasoning Model (https://arxiv.org/pdf/2506.21734) paper.
    
    Args:
        dataset (DatasetDict): The dataset (train, test) to preprocess.
        filter_train (bool, optional): Whether to filter the train set from hard problems. Defaults to False.

    Returns:
        DatasetDict: The preprocessed dataset.
    """
    dataset = load_dataset("sapientinc/sudoku-extreme", cache_dir=cache_dir)
    dataset = dataset.map(lambda x: {"empty_cells": x["question"].count(".")})

    # Create a validation set by stratified sampling on empty_cells
    # Convert 'empty_cells' to ClassLabel for stratification
    dataset = dataset.cast_column("empty_cells", ClassLabel(names=list(map(str, range(82)))))

    train_val_split = dataset["train"].train_test_split(
        test_size=100000, stratify_by_column="empty_cells", seed=42
    )
    
    # Convert 'empty_cells' back to Value (int) for filtering
    train_dataset = train_val_split["train"].cast_column("empty_cells", Value("int64"))
    val_dataset = train_val_split["test"].cast_column("empty_cells", Value("int64"))

    # Filter train set from hard problems
    if filter_train:
        train_dataset = train_dataset.filter(lambda x: x["source"] in ['puzzles0_kaggle', 'puzzles1_unbiased', 'puzzles2_17_clue'])
    
    # Update the dataset with the new split
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": dataset["test"]
    })
    
    return dataset

@register_collate_fn("sudoku")
def collate_fn_sudoku(batch):
    inputs = torch.stack([torch.tensor(list(map(lambda x: 0 if x == '.' else int(x), s['question'])), dtype=torch.float32).reshape(1, 9, 9) for s in batch], dim=0)
    targets = torch.stack([torch.tensor(list(map(lambda x: int(x), s['answer'])), dtype=torch.long).reshape(9, 9) - 1 for s in batch], dim=0) # -1 because the targets are 1-indexed
    metadata = [{'empty_cells': s['empty_cells'], 'rating': s['rating']} for s in batch]
    return inputs, targets, metadata 