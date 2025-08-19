import torch
import numpy as np
import random
import collections.abc
import logging
import os

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

def parse_sweep_args(sweep_args):
    """
    Parses the sweep arguments and returns a dictionary of parameters.
    """
    config = dict()
    sweep_params = [x.split("=") for x in sweep_args]
    for k, v in sweep_params:
        k = k.lstrip('--')
        # Convert the value to the correct type
        if v in ['true', 'false']:
            v = (v == 'true')
        elif v.isdigit():
            v = int(v)
        elif v.replace('.', '', 1).isdigit():
            v = float(v)
        
        keys = k.split('.')
        set_nested_key(config, keys, v)
    return config

def set_nested_key(d, keys, value):
    """
    Set a value in a nested dictionary using a list of keys.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def update_config(config, u):
    """
    Recursively update a dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            config[k] = update_config(config.get(k, {}), v)
        else:
            config[k] = v
    return config 

def get_logger(name):
    logs_dir = os.getenv('LOGS_DIR', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    log_file = os.path.join(logs_dir, f'{name}.log')
    # Delete the log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
    f_handler = logging.FileHandler(log_file)
    c_handler = logging.StreamHandler()
    f_handler.setLevel(logging.INFO)
    c_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)
    
    return logger 

def generate_run_name(config, sweep_args=None):
    """
    Generates a descriptive run name for wandb based on sweep parameters.
    """
    run_name = config['training'].get('experiment_name', 'run')
    if not sweep_args:
        return run_name
    sweep_args = [x.split('=') for x in sweep_args]
    sweep_args = {k.lstrip('--').split('.')[-1]: v for k, v in sweep_args}
    def format_value(v):
        try:
            val = float(v)
            if abs(val) > 0 and abs(val) < 0.01:
                return f"{val:.1e}"
            return f"{val:.4g}"
        except (ValueError, TypeError):
            return str(v)
    param_str = "_".join([f"{k}={format_value(v)}" for k, v in sweep_args.items()])
    return f"{run_name}_{param_str}"

def save_best_model(model, model_dir, eval_results, checkpoint_config, best_metric_value, logger, run):
    """
    Saves the model if the validation metric improves.

    Returns:
        float: The new best metric value.
    """
    metric_name = checkpoint_config['name']
    mode = checkpoint_config['mode']
    current_value = eval_results[metric_name]

    is_better = (mode == 'max' and current_value > best_metric_value) or \
                (mode == 'min' and current_value < best_metric_value)

    if is_better:
        # Format previous value for logging, handling initial -inf
        prev_val_str = f"{best_metric_value:.4f}" if best_metric_value != float('-inf') and best_metric_value != float('inf') else "N/A"
        logger.info(f"New best value for '{metric_name}': {current_value:.4f} (was {prev_val_str}). Saving model.")
        
        best_metric_value = current_value
        save_path = os.path.join(model_dir, "best_model.pt")
        torch.save(model.state_dict(), save_path)
        run.save(save_path)
    
    return best_metric_value