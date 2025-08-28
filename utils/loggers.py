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
        pred_labels = predictions[i].argmax(dim=0).numpy() + 1 # +1 because the predictions are 0-indexed
        target_labels = targets[i].numpy() + 1 # +1 because the targets are 1-indexed

        def grid_to_str(grid, targets=None):
            # Use HTML table for predictions to allow coloring
            html = "<table style='border-collapse: collapse; border: 2px solid black; font-family: monospace;'>"
            
            grid_str = np.vectorize(lambda x: str(x) if x != 0 else '.')(grid.astype(int))
            
            for i in range(9):
                style = "border-top: 1px solid lightgrey;"
                if i % 3 == 0:
                    style="border-top: 2px solid black;"

                html += f"<tr style='{style}'>"
                for j in range(9):
                    cell_style = "width: 1.5em; height: 1.5em; text-align: center; border-left: 1px solid lightgrey;"
                    if j % 3 == 0:
                        cell_style="width: 1.5em; height: 1.5em; text-align: center; border-left: 2px solid black;"

                    cell_value = grid_str[i, j]
                    if targets is not None and cell_value != '.':
                        target_value = str(targets[i, j])
                        if cell_value == target_value:
                            cell_style += " color: green;"
                        else:
                            cell_style += " color: red;"
                    
                    html += f"<td style='{cell_style}'>{cell_value}</td>"
                html += "</tr>"
            html += "</table>"
            return wandb.Html(html)

        table.add_data(
            i,
            grid_to_str(puzzle),
            grid_to_str(pred_labels, targets=target_labels),
            grid_to_str(target_labels)
        )
    
    return table
