import wandb
import torch
import numpy as np

LOGGER_REGISTRY = {}

def register_logger(name):
    def decorator(f):
        LOGGER_REGISTRY[name] = f
        return f
    return decorator

@register_logger("sudoku_logger")
def log_sudoku_predictions_to_wandb(predictions, targets, inputs, max_samples=4):
    """
    Logs Sudoku predictions and targets to a wandb.Table.
    This function should be called within an active wandb run.
    """
    if not wandb.run:
        print("Warning: wandb run not initialized. Skipping logging.")
        return

    # Move tensors to CPU and detach
    predictions = predictions.cpu().detach()
    targets = targets.cpu().detach()
    inputs = inputs.cpu().detach()

    num_samples = min(max_samples, predictions.size(0))
    # # Pick random samples if num_samples is greater than the number of samples
    # if num_samples > predictions.size(0):
    #     indices = np.random.choice(predictions.size(0), num_samples, replace=False)
    #     predictions = predictions[indices]
    #     targets = targets[indices]
    #     inputs = inputs[indices]

    if num_samples == 0:
        return

    columns = ["ID", "Input Puzzle", "Predicted Solution", "Ground Truth"]
    table = wandb.Table(columns=columns)

    for i in range(num_samples):
        puzzle = inputs[i].squeeze(0).numpy()
        pred_labels = predictions[i].argmax(dim=1).numpy() + 1 # +1 because the predictions are 0-indexed
        target_labels = targets[i].numpy() + 1 # +1 because the targets are 1-indexed

        def grid_to_str(grid):
            # Replace 0 with '.' for empty cells for better readability
            grid_str = np.vectorize(lambda x: str(x) if x != 0 else '.')(grid.astype(int))
            return '\n'.join([' '.join(row) for row in grid_str])

        table.add_data(
            i,
            grid_to_str(puzzle),
            grid_to_str(pred_labels),
            grid_to_str(target_labels)
        )
    
    return table
