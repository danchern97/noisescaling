import torch
from datasets import load_dataset, DatasetDict

########################################################
# SUDOKU
########################################################

def load_and_preprocess_sudoku(cache_dir, filter_train : int = 55) -> DatasetDict:
    """
    Preprocesses the sudoku dataset.

    Args:
        dataset (DatasetDict): The dataset (train, test) to preprocess.
        filter_train (int, optional): The maximum number of empty cells in the train set to limit the difficulty and study easy-to-hard scaling. Defaults to None.

    Returns:
        DatasetDict: The preprocessed dataset.
    """
    dataset = load_dataset("sapientinc/sudoku-extreme", cache_dir=cache_dir)
    dataset = dataset.map(lambda x: {"empty_cells": x["question"].count(".")})
    if filter_train is not None:
        dataset["train"] = dataset["train"].filter(lambda x: x["empty_cells"] <= filter_train)
    return dataset

def collate_fn_sudoku(batch):
    questions = torch.stack([torch.tensor(list(map(lambda x: 0 if x == '.' else int(x), s['question']))).reshape(9, 9) for s in batch], dim=0)
    answers = torch.stack([torch.tensor(list(map(lambda x: int(x), s['answer']))).reshape(9, 9) for s in batch], dim=0)
    metadata = [{'empty_cells': s['empty_cells'], 'rating': s['rating']} for s in batch]
    return questions, answers, metadata
