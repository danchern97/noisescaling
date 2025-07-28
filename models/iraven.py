import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from . import register_model, register_loss


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=8, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Remove softmax to return logits


@register_loss("cross_entropy")
def get_cross_entropy_loss(predictions, targets, **kwargs):
    """
    Computes cross entropy loss for classification.
    
    Args:
        predictions (torch.Tensor): Model predictions (logits)
        targets (torch.Tensor): Target labels
        
    Returns:
        torch.Tensor: Cross entropy loss
    """
    return F.cross_entropy(predictions, targets)


@register_model("iraven")
class IRavenModel(nn.Module):
    def __init__(self, pretrained=True, hidden_dim=128, dropout=0.5, scaler=None, aggregator=None, **kwargs):
        super().__init__()
        # Use ResNet with embedding output (no classification layer)
        self.resnet = resnet18(pretrained=pretrained, return_embedding=True)
        
        # Calculate input dimension for MLP based on ResNet output + context structure
        # Context: (2, 3) = 6 images, Pairs: 2 images, Answers: 8 images = 16 total images
        # Each image gives 512 features (ResNet-18 embedding dimension)
        total_images = 6 + 2 + 8  # context + pairs + possible_answers
        resnet_embedding_dim = 512  # ResNet-18 embedding dimension
        mlp_input_dim = total_images * resnet_embedding_dim
        
        self.mlp = SimpleMLP(input_dim=mlp_input_dim, hidden_dim=hidden_dim, output_dim=8, dropout=dropout)

    def forward(self, inputs):
        """
        Forward pass for I-RAVEN model.
        
        Args:
            inputs: tuple of (context, pairs, possible_answers) or individual tensors
                context: (B, 2, 3, C, H, W) - context images
                pairs: (B, 2, C, H, W) - pair images  
                possible_answers: (B, 8, C, H, W) - possible answer images
            
        Returns:
            logits: (B, 8) - logits for 8 possible answers
        """
        # Handle both tuple input (from training infrastructure) and individual arguments
        if isinstance(inputs, tuple) and len(inputs) == 3:
            context, pairs, possible_answers = inputs
        else:
            # For backward compatibility or direct usage
            context, pairs, possible_answers = inputs
        # Get batch size
        B = context.shape[0]
        
        # Flatten context images and get embeddings
        B, N, T, C, H, W = context.shape
        context_flat = context.view(B * N * T, C, H, W)
        context_emb = self.resnet(context_flat)
        
        # Flatten pairs and get embeddings
        B, N, C, H, W = pairs.shape
        pairs_flat = pairs.view(B * N, C, H, W)
        pairs_emb = self.resnet(pairs_flat)
        
        # Flatten possible answers and get embeddings
        B, N, C, H, W = possible_answers.shape
        answers_flat = possible_answers.view(B * N, C, H, W)
        answers_emb = self.resnet(answers_flat)
        
        # Get embedding dimension
        emb_dim = context_emb.shape[-1]
        
        # Reshape embeddings back to batch structure
        context_emb = context_emb.view(B, -1, emb_dim)      # (B, 6, emb_dim)
        pairs_emb = pairs_emb.view(B, -1, emb_dim)          # (B, 2, emb_dim)  
        answers_emb = answers_emb.view(B, -1, emb_dim)      # (B, 8, emb_dim)
        
        # Concatenate all embeddings
        all_emb = torch.cat([context_emb, pairs_emb, answers_emb], dim=1)  # (B, 16, emb_dim)
        
        # Flatten for MLP
        input_flattened = all_emb.view(B, -1)  # (B, 16 * emb_dim)
        
        # Get logits from MLP
        logits = self.mlp(input_flattened)  # (B, 8)
        
        return logits



