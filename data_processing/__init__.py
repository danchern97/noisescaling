import torch
from datasets import load_dataset, DatasetDict, ClassLabel, Value
from torch.utils.data import Subset
from torchvision import transforms


# Make upstream RAVEN modules importable regardless of CWD by adding their path
import sys
from pathlib import Path
_this = Path(__file__).resolve()
_arc_root = _this.parents[2]
_raven_model_dir = _arc_root / 'RAVEN' / 'src' / 'model'
if str(_raven_model_dir) not in sys.path:
    sys.path.insert(0, str(_raven_model_dir))

# Try importing upstream dataset utilities
try:
    from utility.dataset_utility import dataset as RAVENUpstreamDataset  # type: ignore
    from utility.dataset_utility import ToTensor as RAVENToTensor  # type: ignore
except Exception:
    RAVENUpstreamDataset = None
    RAVENToTensor = None


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
        tuple: (inputs, targets, metadata) - compatible with training infrastructure
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
    targets_batch = torch.tensor(targets, dtype=torch.long)  # Shape: (batch_size,)
    
    # Create metadata
    metadata = [{
        'predict': sample['predict'],
        'meta_structure': sample['meta_structure'],
        'meta_matrix': sample['meta_matrix'],
        'meta_target': sample['meta_target']
    } for sample in batch]
    
    # Package inputs as a tuple that the IRavenModel.forward() expects
    inputs = (context_batch, pairs_batch, answers_batch)
    
    return inputs, targets_batch, metadata 


# -------------------------------
# RAVEN (authors' format) dataset
# -------------------------------

@register_collate_fn("raven")
def collate_fn_raven(batch):
    """
    Collate function for the upstream RAVEN dataset.

    Each sample is a tuple: (images, target, meta_target, meta_structure, embedding, indicator)
    - images: Tensor of shape (16, H, W)
    - target: int label in [0, 7]
    - embedding: Tensor of shape (6, 300)
    - indicator: Tensor of shape (1,)
    We assemble inputs as a tuple (images, embedding, indicator) to be unpacked by the model.
    """
    images = torch.stack([sample[0] for sample in batch], dim=0)           # (B, 16, H, W)
    targets = torch.tensor([int(sample[1]) for sample in batch], dtype=torch.long)
    meta_targets = [sample[2] for sample in batch]
    meta_structures = [sample[3] for sample in batch]
    embeddings = torch.stack([sample[4] for sample in batch], dim=0)       # (B, 6, 300)
    indicators = torch.stack([sample[5] for sample in batch], dim=0)       # (B, 1)

    inputs = (images, embeddings, indicators)
    metadata = {
        'meta_target': meta_targets,
        'meta_structure': meta_structures,
    }
    return inputs, targets, metadata


@register_dataset("raven")
def load_raven(cache_dir, img_size: int = 224, **kwargs) -> DatasetDict:
    """
    Loads the upstream RAVEN dataset using the authors' dataset utility.

    Args:
        cache_dir: Root directory of RAVEN where subfolders per configuration exist and `embedding.npy` is present.
        img_size: Target image size (resizing is applied by the upstream dataset loader).
    Returns:
        DatasetDict with `train`, `validation`, `test` splits.
    """
    if RAVENUpstreamDataset is None:
        raise ImportError("RAVEN upstream dataset code is not available. Ensure the RAVEN repo exists and is importable.")

    # The upstream dataset expects a `transform` that converts numpy arrays to tensors
    transform = transforms.Compose([RAVENToTensor()]) if RAVENToTensor is not None else None

    # Initialize upstream datasets for each split. The upstream class filters by filename containing the split name.
    train = RAVENUpstreamDataset(cache_dir, "train", img_size, transform=transform)
    val = RAVENUpstreamDataset(cache_dir, "val", img_size, transform=transform)
    test = RAVENUpstreamDataset(cache_dir, "test", img_size, transform=transform)

    return DatasetDict({
        'train': train,
        'validation': val,
        'test': test,
    })


@register_dataset("iraven")
def load_iraven(cache_dir, data_dir=None, transform=None, **kwargs) -> DatasetDict:
    """
    Loads the I-RAVEN dataset from .npz files.
    
    Args:
        cache_dir (str): Path to directory containing .npz files (used as data_dir for I-RAVEN)
        data_dir (str, optional): Alternative path specification (if provided, overrides cache_dir)
        transform (callable, optional): Optional transform to be applied to images
    
    Returns:
        DatasetDict: Dictionary containing train, validation, and test datasets
    """
    # For I-RAVEN, use cache_dir as the data directory unless data_dir is explicitly provided
    if data_dir is None:
        data_dir = cache_dir
    
    # Define a simple preprocessing transform if none provided
    if transform is None:
        # Import transforms here to avoid circular imports        
        # Use the same preprocessing as in the notebook
        def preprocess_image(image):
            # Ensure image is in the right format for ToPILImage
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            
            # Apply the transform pipeline
            transform_pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),  # ensure 3 channels
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            return transform_pipeline(image)
        
        transform = preprocess_image
    
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

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    return dataset_dict