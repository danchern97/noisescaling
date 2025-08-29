import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import torch.nn.functional as F
import yaml
import argparse
import wandb
import os
from dotenv import load_dotenv
import numpy as np
from torchvision.utils import make_grid
from collections import OrderedDict
from huggingface_hub import hf_hub_download

from data_processing import DATASET_REGISTRY, get_collate_fn
from models import MODEL_REGISTRY, LOSS_REGISTRY
from utils import count_trainable_parameters, set_seed
from utils.logger import get_logger
from utils.metrics import METRIC_REGISTRY
from tqdm.auto import tqdm
import glob

def cleanup_checkpoints(model_dir, logger, keep_last=5):
    checkpoints = sorted(
        glob.glob(os.path.join(model_dir, "model_*.pt")),
        key=os.path.getmtime
    )
    if len(checkpoints) > keep_last:
        for ckpt in checkpoints[:-keep_last]:
            try:
                os.remove(ckpt)
                logger.info(f"Deleted old checkpoint: {ckpt}")
            except OSError as e:
                logger.warning(f"Error deleting {ckpt}: {e}")

def get_model_by_name(model_name : str, **kwargs):
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)

def get_dataloaders(config):
    dataset_fn = DATASET_REGISTRY[config['dataset']['name']]
    dataset = dataset_fn(cache_dir=config['dataset']['path'])
    collate_fn = get_collate_fn(config['dataset']['name'])
    dataloaders = {}
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            dataloaders[split] = DataLoader(
                dataset[split],
                batch_size=config['training']['batch_size'],
                shuffle=True if split == 'train' else False,
                num_workers=config['training']['num_workers'],
                pin_memory=True,
                collate_fn=collate_fn
            )
    return dataloaders

def run_metrics(predictions, targets, model, inputs, device, metrics, results=None):
    if results is None:
        results = {metric: 0.0 for metric in metrics}
    for metric in metrics:
        results[metric] += METRIC_REGISTRY[metric](predictions=predictions, targets=targets, model=model, inputs=inputs, device=device).item()
    return results

def load_state_dict(path, version=None):
    
    saved_state_dict = torch.load(path)

    if version:
        
        new_state_dict = OrderedDict()

        if version == 'baseline':

            for old_key, value in saved_state_dict.items():
                
                new_key = old_key

                # --- CORE RENAMING LOGIC for the 'mid' layers ---
                if old_key.startswith('mid.0.'):
                    # Replace 'mid.0.' with 'mid_1.'
                    new_key = old_key.replace('mid.0.', 'mid_1.', 1)
                elif old_key.startswith('mid.1.'):
                    # Replace 'mid.1.' with 'mid_2.'
                    new_key = old_key.replace('mid.1.', 'mid_2.', 1)
                elif old_key.startswith('mid.2.'):
                    # Replace 'mid.2.' with 'mid_3.'
                    new_key = old_key.replace('mid.2.', 'mid_3.', 1)
                elif old_key.startswith('mid.3.'):
                    # Replace 'mid.3.' with 'mid_4.'
                    new_key = old_key.replace('mid.3.', 'mid_4.', 1)

                print(f"Mapping '{old_key}' to '{new_key}'")
                new_state_dict[new_key] = value

        else:
            raise ValueError(f"Version {version} not implemented to load state dict.")

        saved_state_dict = new_state_dict

    return saved_state_dict

def eval_model(model, dataloader, loss_fns, device, metrics):
    model.eval()
    results = {metric: 0.0 for metric in metrics}
    results['loss'] = 0.0
    for loss in loss_fns:
        results[loss['name']] = 0.0
    
    for inputs, targets, metadata in dataloader:
        with torch.no_grad():
            # if inputs is a tuple pass all of them to device
            if isinstance(inputs, tuple):
                inputs = tuple(inp.to(device) for inp in inputs)
            elif isinstance(inputs, list):
                inputs = [inp.to(device) for inp in inputs]
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)
            model_output = model(inputs)
            
            if isinstance(model_output, dict):
                predictions = model_output['predictions']
                expert_reps = model_output.get('expert_representations', None)
            else:
                predictions = model_output
                expert_reps = None

            loss_values = torch.tensor([loss['fn'](predictions, targets, **({"expert_reps": expert_reps})) * loss['weight'] for loss in loss_fns])
            results['loss'] += torch.sum(loss_values).item()
            results = run_metrics(predictions, targets, model, inputs, device, metrics, results)
            for i, loss in enumerate(loss_fns):
                results[loss['name']] += loss_values[i].item()

    for metric in results:
        results[metric] /= len(dataloader)
    return results


_LOGGED_STATIC_RAVEN_IMAGES = False
_RAVEN_LOGITS_COLUMNS = None
_RAVEN_LOGITS_ROWS = []


def _prepare_raven_example(inputs, predictions, targets):
    """
    Build a W&B-friendly example from a single RAVEN batch.

    Inputs shape (tuple):
      - images: (B, 16, H, W) grayscale in [0,1] or arbitrary
      - embedding: (B, 6, 300)
      - indicator: (B, 1)
    predictions: (B, 8) logits
    targets: (B,) long

    Returns a dict with a wandb.Image grid and a table of class probabilities/logits.
    """
    try:
        images, embedding, indicator = inputs
    except Exception:
        return None

    if not (isinstance(images, torch.Tensor) and images.dim() == 4 and images.size(1) == 16):
        return None

    # Pick first example in batch
    img_16 = images[0]  # (16, H, W)
    target = targets[0].item() if isinstance(targets, torch.Tensor) else int(targets[0])
    logits = predictions[0].detach().float().cpu()  # (8,)
    probs = torch.softmax(logits, dim=-1)

    # Panels: first 8 are context (3x3 with one missing), last 8 are answer choices
    ctx = img_16[:8]   # (8, H, W)
    ans = img_16[8:]   # (8, H, W)

    def to_three_channel(t_single: torch.Tensor) -> torch.Tensor:
        # t_single: (H,W) or (C,H,W)
        if t_single.dim() == 2:
            t_single = t_single.unsqueeze(0)
        if t_single.size(0) != 1 and t_single.size(0) != 3:
            t_single = t_single.mean(dim=0, keepdim=True)
        # Min-max normalize
        s_min = t_single.amin(dim=(-2, -1), keepdim=True)
        s_max = t_single.amax(dim=(-2, -1), keepdim=True)
        t_single = (t_single - s_min) / (s_max - s_min + 1e-8)
        if t_single.size(0) == 1:
            t_single = t_single.repeat(3, 1, 1)
        return t_single

    def add_border(img: torch.Tensor, border: int = 6, color: float = 1.0) -> torch.Tensor:
        # img: (3,H,W), add constant border
        return F.pad(img, (border, border, border, border), value=color)

    # Build 3x3 context grid with last tile blank and thick borders
    ctx_tiles = [add_border(to_three_channel(ctx[i])) for i in range(8)]
    c, h, w = ctx_tiles[0].shape
    blank = add_border(torch.ones(3, h - 12, w - 12)) if (h >= 12 and w >= 12) else add_border(torch.ones(3, max(1, h - 12), max(1, w - 12)))
    # Recompute shapes after padding
    c, h, w = ctx_tiles[0].shape
    row1 = torch.cat(ctx_tiles[0:3], dim=2)
    row2 = torch.cat(ctx_tiles[3:6], dim=2)
    row3 = torch.cat([ctx_tiles[6], ctx_tiles[7], blank], dim=2)
    ctx_grid = torch.cat([row1, row2, row3], dim=1)

    # Build answers grid 2x4 with thick borders
    ans_tiles = [add_border(to_three_channel(ans[i])) for i in range(8)]
    row_a1 = torch.cat(ans_tiles[0:4], dim=2)
    row_a2 = torch.cat(ans_tiles[4:8], dim=2)
    ans_grid = torch.cat([row_a1, row_a2], dim=1)

    return {
        "context_grid": wandb.Image(ctx_grid.detach().cpu().permute(1, 2, 0).numpy(), caption="context 3x3 (last empty)"),
        "answers_grid": wandb.Image(ans_grid.detach().cpu().permute(1, 2, 0).numpy(), caption="answer choices (8)"),
        "target": target,
        "pred": int(torch.argmax(probs).item()),
        "logits": logits.tolist(),
        "probs": probs.tolist(),
    }


def log_validation_example(run, model, dataloader, device, step: int, prefix: str = "val"):
    """Log a single RAVEN example (context + answers + probs) to W&B if compatible."""
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return
    except Exception:
        return

    inputs, targets, _ = batch
    # Move to device like training loop
    if isinstance(inputs, tuple):
        inputs_dev = tuple(inp.to(device) for inp in inputs)
    elif isinstance(inputs, list):
        inputs_dev = [inp.to(device) for inp in inputs]
    else:
        inputs_dev = inputs.to(device)
    targets_dev = targets.to(device) if isinstance(targets, torch.Tensor) else targets

    model.eval()
    with torch.no_grad():
        model_output = model(inputs_dev)
        predictions = model_output['predictions'] if isinstance(model_output, dict) else model_output

    example = _prepare_raven_example(inputs, predictions, targets_dev)
    if example is None:
        return
    global _LOGGED_STATIC_RAVEN_IMAGES, _RAVEN_LOGITS_COLUMNS, _RAVEN_LOGITS_ROWS
    if not _LOGGED_STATIC_RAVEN_IMAGES:
        run.log({
            f"{prefix}/example_context": example["context_grid"],
            f"{prefix}/example_answers": example["answers_grid"],
        })
        _LOGGED_STATIC_RAVEN_IMAGES = True

    # Maintain a single W&B table of logits over validation steps by rebuilding from stored rows
    if _RAVEN_LOGITS_COLUMNS is None:
        _RAVEN_LOGITS_COLUMNS = ["step"] + [f"logit_{i}" for i in range(8)]
    _RAVEN_LOGITS_ROWS.append([int(step)] + [float(x) for x in example["logits"]])
    table = wandb.Table(data=_RAVEN_LOGITS_ROWS, columns=_RAVEN_LOGITS_COLUMNS)
    run.log({f"{prefix}/example_logits_table": table})

def train_model(config):
    # Load environment variables
    load_dotenv(dotenv_path='.env')

    # Set seed for reproducibility
    set_seed(config['training']['seed'])

    # Create logger
    logger = get_logger(config['training']['experiment_name'])

    # Create model checkpoint directory
    base_model_dir = os.getenv('MODELS_DIR', 'models_cache')
    model_dir = os.path.join(base_model_dir, config['training']['experiment_name'])
    os.makedirs(model_dir, exist_ok=True)

    # Initialize wandb
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    run = wandb.init(
        project=os.getenv('WANDB_PROJECT'),
        entity=os.getenv('WANDB_ENTITY'),
        name=config['training']['experiment_name'],
        config=config
    )

    # Set up the scaler and aggregator
    scaler, aggregator = None, None
    model_config = config['model']
    injection_point = model_config.get('injection_point')


    if model_config.get('scaler', None):
        scaler_config = model_config['scaler']
        scaler_args = scaler_config.get('args', {})

        if injection_point is not None:
            if injection_point == '0':
                scaler_args['layer_type'] = 'conv'
                scaler_args['dim'] = 256  # number of channels
            elif injection_point == '1':
                scaler_args['layer_type'] = 'conv'
                scaler_args['dim'] = 512
            elif injection_point == '2':
                scaler_args['layer_type'] = 'conv'
                scaler_args['dim'] = 512
            elif injection_point == '3':
                scaler_args['layer_type'] = 'conv'
                scaler_args['dim'] = 1024
            elif injection_point == '4':
                scaler_args['layer_type'] = 'conv'
                scaler_args['dim'] = 9

            else:
                raise ValueError(f"Unsupported injection_point: {injection_point}")
            
            logger.info(f"Using scaler at injection point: {injection_point} with args: {scaler_args}")
            scaler = get_model_by_name(scaler_config['name'], **scaler_args)
        else:
            # Instantiate scaler without injection-specific args (e.g., for RAVEN embedding scaler)
            scaler = get_model_by_name(scaler_config['name'], **scaler_args)

    if model_config.get('aggregator', None):
        aggregator = get_model_by_name(model_config['aggregator']['name'], **model_config['aggregator']['args'])

    # Get the model from the registry
    model = get_model_by_name(model_config['name'], scaler=scaler, aggregator=aggregator, injection_point=injection_point, dropout=model_config.get('args', {}).get('dropout', 0.0))
    logger.info(f"Model Architecture:\n{model}")
    # Load pretrained weights, if provided.
    if model_config.get('pretrained_path', None):
        
        if model_config.get('pretrained_from_hf', None):

            pretrained_path = hf_hub_download(repo_id=model_config['pretrained_from_hf'], filename=model_config['pretrained_path'])
            logger.info(f"Loading pretrained weights from Hugging Face Hub: {pretrained_path}")
        else:
            pretrained_path = model_config['pretrained_path']

        logger.info(f"Loading pretrained weights from {pretrained_path}")
        state_dict = load_state_dict(pretrained_path, version=model_config.get('pretrained_version'))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        
    run.watch(model)
    
    num_trainable_params = count_trainable_parameters(model)
    logger.info(f"Number of trainable parameters: {num_trainable_params}")
    run.config.update({"num_trainable_params": num_trainable_params})

    device = torch.device(config['training'].get('device', 'cuda:0') if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the dataloaders
    dataloaders = get_dataloaders(config)
    
    # Set up the optimizer
    if config['training'].get('scaler_regime', None) == 'full' or config['training'].get('scaler_regime', None) is None:
        parameters = model.parameters()
    elif config['training'].get('scaler_regime', None) == 'partial':
        if getattr(model, 'scaler', None) is None or getattr(model, 'aggregator', None) is None:
            raise ValueError("scaler_regime 'partial' requires both model.scaler and model.aggregator to be set. Check your config.")
        parameters = list(model.scaler.parameters()) + list(model.aggregator.parameters())

    else:
        raise ValueError(f"Invalid scaler regime: {config['training']['scaler_regime']}. Supported regimes are 'full' and 'partial'.")
    optimizer = AdamW(parameters, lr=config['training'].get('learning_rate', 0.001))
    scheduler_config = config['training'].get('scheduler', None)
    scheduler = None
    if scheduler_config:
        if scheduler_config['name'] == 'linear':
            scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=scheduler_config['steps'])
        else:
            raise ValueError(f"Scheduler {scheduler_config['name']} not supported.")
    # Get the loss function from the registry
    loss_fns = [{'name': loss['name'], 'fn': LOSS_REGISTRY[loss['name']], 'weight': loss['weight']} for loss in model_config['losses']]
    model.compile()

    eval_results = eval_model(model, dataloaders['validation'], loss_fns, device, config['training']['validation_metrics'])
    log_msg = f"Validation results at the beginning: "
    log_msg += ", ".join([f"{k}: {v:.4f}" for k, v in eval_results.items()])
    logger.info(log_msg)
    log_dict = {f"val/{k}": v for k, v in eval_results.items()}
    run.log(log_dict)
    # Also log a qualitative example at the beginning
    if 'validation' in dataloaders:
        log_validation_example(run, model, dataloaders['validation'], device, step=0, prefix="val")


    model.train()
    step = 0
    for epoch in range(config['training'].get('epochs', 100)):
        for _, (inputs, targets, _) in enumerate(tqdm(dataloaders['train'], desc='Training')):
            
            # 2. THE FIX: If in partial regime, selectively set baseline modules to eval mode
            if config['training'].get('scaler_regime', None) == 'partial':
                if hasattr(model, 'enc'): model.enc.eval()
                if hasattr(model, 'mid_1'): model.mid_1.eval()
                if hasattr(model, 'mid_2'): model.mid_2.eval()
                if hasattr(model, 'mid_3'): model.mid_3.eval()
                if hasattr(model, 'mid_4'): model.mid_4.eval()
                if hasattr(model, 'dec'): model.dec.eval()

            # if inputs is a tuple pass all of them to device
            if isinstance(inputs, tuple):
                inputs = tuple(inp.to(device) for inp in inputs)
            elif isinstance(inputs, list):
                inputs = [inp.to(device) for inp in inputs]
            else:
                inputs = inputs.to(device)

            targets = targets.to(device)
                    
            model_output = model(inputs)
            #predictions = model(inputs)

            if isinstance(model_output, dict):
                predictions = model_output['predictions']
                expert_reps = model_output.get('expert_representations', None)
            else:
                predictions = model_output
                expert_reps = None

            # Multiple losses are summed up with corresponding coefficients
            loss_values = torch.stack([loss['fn'](predictions, targets, **({"expert_reps": expert_reps})) * loss['weight'] for loss in loss_fns], dim=0)
            loss = torch.sum(loss_values)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler:
                scheduler.step()

            log_dict = run_metrics(predictions, targets, model, inputs, device, config['training']['training_metrics'])
            for i, loss_fn in enumerate(loss_fns):
                log_dict[loss_fn['name']] = loss_values[i].item()
            log_dict['loss'] = loss.item()
            if scheduler:
                log_dict['lr'] = scheduler.get_last_lr()[0]
            else:
                log_dict['lr'] = optimizer.param_groups[0]['lr']

            log_dict = {f"train/{k}": v for k, v in log_dict.items()}
            run.log(log_dict)
            step += 1

            if step % config['training'].get('eval_interval', 500) == 0 and 'validation' in dataloaders:
                eval_results = eval_model(model, dataloaders['validation'], loss_fns, device, config['training']['validation_metrics'])
                log_msg = f"Validation results at epoch {epoch}, step {step}: "
                log_msg += ", ".join([f"{k}: {v:.4f}" for k, v in eval_results.items()])
                logger.info(log_msg)
                log_dict = {f"val/{k}": v for k, v in eval_results.items()}
                run.log(log_dict)
                # Log a qualitative example periodically
                log_validation_example(run, model, dataloaders['validation'], device, step=step, prefix="val")
                
                # Save model checkpoint
                model_path = os.path.join(model_dir, f"model_{step}.pt")
                torch.save(model.state_dict(), model_path)
                run.save(model_path)
                # Delete old checkpoints
                cleanup_checkpoints(model_dir, logger, keep_last=5)

                model.train()

    # Evaluate the final model on the test set
    if 'test' in dataloaders:
        eval_results = eval_model(model, dataloaders['test'], loss_fns, device, config['training']['validation_metrics'])
        logger.info(f"Test results: {eval_results}")
        eval_results = {f"test/{k}": v for k, v in eval_results.items()}
        run.log(eval_results)

    # Finish the wandb run
    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_model(config)
