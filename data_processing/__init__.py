import torch
from datasets import load_dataset, DatasetDict, ClassLabel, Value
from torch.utils.data import Subset
from iravendataset import IRavenDataset

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
def load_sudoku(cache_dir, filter_train : int = 55) -> DatasetDict:
    """
    Preprocesses the Sudoku (https://huggingface.co/datasets/sapientinc/sudoku-extreme) dataset.

    The dataset is used in the Hierarchical Reasoning Model (https://arxiv.org/pdf/2506.21734) paper.
    
    Args:
        dataset (DatasetDict): The dataset (train, test) to preprocess.
        filter_train (int, optional): The maximum number of empty cells in the train set to limit the difficulty and study easy-to-hard scaling. Defaults to None.

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
    if filter_train is not None:
        train_dataset = train_dataset.filter(lambda x: x["empty_cells"] <= filter_train)
    
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
    targets = torch.stack([torch.tensor(list(map(lambda x: int(x), s['answer']))).reshape(9, 9) - 1 for s in batch], dim=0) # -1 because the targets are 1-indexed
    metadata = [{'empty_cells': s['empty_cells'], 'rating': s['rating']} for s in batch]
    return inputs, targets, metadata


@register_collate_fn("iraven")
def collate_fn_iraven(batch):
    """
    Collate function for I-RAVEN dataset.
    
    Args:
        batch: List of samples from IRavenDataset
        
    Returns:
        tuple: (context, pairs, possible_answers, targets, metadata)
    """
    # Extract all components from the batch
    contexts = [sample['context'] for sample in batch]
    pairs = [sample['pairs'] for sample in batch]
    possible_answers = [sample['possible_answers'] for sample in batch]
    targets = [sample['target'] for sample in batch]
    
    # Stack tensors
    context_batch = torch.stack(contexts)  # Shape: (batch_size, 2, 3, H, W)
    pairs_batch = torch.stack(pairs)       # Shape: (batch_size, 2, H, W)
    answers_batch = torch.stack(possible_answers)  # Shape: (batch_size, 8, H, W)
    targets_batch = torch.tensor(targets)  # Shape: (batch_size,)
    
    # Create metadata
    metadata = [{
        'predict': sample['predict'],
        'meta_structure': sample['meta_structure'],
        'meta_matrix': sample['meta_matrix'],
        'meta_target': sample['meta_target']
    } for sample in batch]
    
    return context_batch, pairs_batch, answers_batch, targets_batch, metadata 


@register_dataset("iraven")
def load_iraven(data_dir=None, transform=None) -> DatasetDict:
    """
    Loads the I-RAVEN dataset from .npz files.
    
    Args:
        cache_dir (str): Cache directory (not used for I-RAVEN, kept for consistency)
        data_dir (str): Path to directory containing .npz files (e.g., 'data/center_single')
        transform (callable, optional): Optional transform to be applied to images
    
    Returns:
        DatasetDict: Dictionary containing train, validation, and test datasets
    """
    if data_dir is None:
        raise ValueError("data_dir must be specified for I-RAVEN dataset")
    

    dataset = IRavenDataset(data_dir, transform=transform)
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Create indices for splitting
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create DatasetDict

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    return dataset_dict