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
            # Keep images as numpy arrays for torchvision transforms
            images = npz['image']  # Get all images
            
            # Context: images 0-5 arranged as (2, 3, H, W)
            context = np.array([
                [images[i], images[i+1], images[i+2]] for i in range(0, 6, 3)
            ])  # shape: (2, 3, H, W)
            
            # Pairs: images 6-7 
            pairs = np.array([images[6], images[7]])  # shape: (2, H, W)
            
            # Possible answers: images 8-15
            possible_answers = np.array([images[i] for i in range(8, 16)])  # shape: (8, H, W)

            self.samples.append({
                'context': context,
                'pairs': pairs,
                'possible_answers': possible_answers,
                'target': int(npz['target']),
                'predict': int(npz['predict']),
                'meta_structure': npz['meta_structure'],
                'meta_matrix': npz['meta_matrix'],
                'meta_target': npz['meta_target'],
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

        # Apply transform to all images (transform should return tensors)
        if self.transform:
            # Apply transform to each image individually
            # Context: (2, 3, H, W) -> apply to each of the 6 images
            context_list = []
            for i in range(context.shape[0]):
                triplet_list = []
                for j in range(context.shape[1]):
                    # Transform expects numpy array, returns tensor
                    transformed_img = self.transform(context[i, j])
                    triplet_list.append(transformed_img)
                context_list.append(torch.stack(triplet_list))
            context = torch.stack(context_list)
            
            # Pairs: (2, H, W) -> apply to each of the 2 images
            pairs_list = []
            for i in range(pairs.shape[0]):
                transformed_img = self.transform(pairs[i])
                pairs_list.append(transformed_img)
            pairs = torch.stack(pairs_list)
            
            # Possible answers: (8, H, W) -> apply to each of the 8 images
            answers_list = []
            for i in range(possible_answers.shape[0]):
                transformed_img = self.transform(possible_answers[i])
                answers_list.append(transformed_img)
            possible_answers = torch.stack(answers_list)
        else:
            # If no transform, convert to tensors manually
            context = torch.from_numpy(context).float()
            pairs = torch.from_numpy(pairs).float()
            possible_answers = torch.from_numpy(possible_answers).float()

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