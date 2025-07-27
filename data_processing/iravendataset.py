import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IRavenDataset(Dataset):
    """
    PyTorch Dataset for loading I-RAVEN .npz files from a directory.
    Each .npz file is expected to contain keys: 'image', 'target', etc.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to directory containing .npz files.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., preprocessing for images).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.samples = []
        for file in self.file_list:
            npz = np.load(file)

            # pairs are 0-2, 3-5, 6-7 are context, 8 is predicted, so 8-15 are possible answers
            # Use np.array to avoid UserWarning about slow tensor creation from list of ndarrays
            context_np = np.array([
                [npz['image'][i], npz['image'][i+1], npz['image'][i+2]] for i in range(0, 6, 3)
            ])  # shape: (2, 3, H, W)
            context = torch.tensor(context_np)
            pairs = npz['image'][6:8]
            possible_answers = npz['image'][8:]

            self.samples.append({
                'context': context,
                'pairs': pairs,
                'possible_answers': possible_answers,
                'target': npz['target'],
                'predict': npz['predict'],
                'meta_structure': npz['meta_structure'],
                'meta_matrix': npz['meta_matrix'],
                'meta_target': npz['meta_target'],
                'meta_structure': npz['meta_structure'],
            })  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        context = sample['context']
        pairs = sample['pairs']
        possible_answers = sample['possible_answers']
        target = sample['target']
        predict = sample['predict']
        meta_structure = sample['meta_structure']
        meta_matrix = sample['meta_matrix']
        meta_target = sample['meta_target']

        # Optionally apply transform to all images
        if self.transform:
            context = torch.stack([torch.stack([self.transform(img) for img in triplet]) for triplet in context])
            pairs = torch.stack([self.transform(img) for img in pairs])
            possible_answers = torch.stack([self.transform(img) for img in possible_answers])

        return {
            'context': context,
            'pairs': pairs,
            'possible_answers': possible_answers,
            'target': target,
            'predict': predict,
            'meta_structure': meta_structure,
            'meta_matrix': meta_matrix,
            'meta_target': meta_target
        }