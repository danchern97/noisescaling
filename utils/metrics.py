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

@register_metric("accuracy_unknown_cells")
def get_accuracy_unknown_cells(predictions, targets, inputs, dim=1, **kwargs):
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