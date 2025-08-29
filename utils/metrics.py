import torch

METRIC_REGISTRY = {}

def register_metric(name):
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator

@register_metric("accuracy")
def get_accuracy(predictions, targets, dim=1, **kwargs):
    """
    Computes the accuracy of the predictions.

    Args:
        predictions (torch.Tensor): The predictions.
        targets (torch.Tensor): The targets.
        dim (int): The dimension to reduce.
        **kwargs: Additional keyword arguments.
    """
    return (predictions.argmax(dim=dim) == targets).float().mean() 

@register_metric("accuracy_sudoku")
def get_accuracy_sudoku(predictions, targets, inputs, dim=1, **kwargs):
    """
    Computes the accuracy of the unknown sudoku cells.

    Args:
        predictions (torch.Tensor): The logits of the predictions, shape (B, 9, 9, 9).
        targets (torch.Tensor): The targets, shape (B, 9, 9).
        inputs (torch.Tensor): The inputs, shape (B, 9, 9).
        dim (int): The dimension of logits to reduce.
        **kwargs: Additional keyword arguments.
    """
    unknown_mask = (inputs == 0).squeeze(dim)
    return (predictions.argmax(dim=dim)[unknown_mask] == targets[unknown_mask]).float().mean()


@register_metric("maze_iou")
def get_maze_iou(predictions, targets, threshold: float = 0.5, **kwargs):
    """
    Intersection over Union for binary mask predictions.
    Expects raw logits in predictions of shape (B, 1, H, W).
    """
    probs = torch.sigmoid(predictions)
    preds_bin = (probs > threshold).float()
    targets_bin = (targets > 0.5).float()
    intersection = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
    union = ((preds_bin + targets_bin) > 0).float().sum(dim=(1, 2, 3)) + 1e-6
    return (intersection / union).mean()