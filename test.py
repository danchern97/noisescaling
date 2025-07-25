import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import yaml
import argparse
import wandb
import os
from dotenv import load_dotenv

from data_processing import DATASET_REGISTRY, get_collate_fn
from models import MODEL_REGISTRY, LOSS_REGISTRY
from utils import count_trainable_parameters, set_seed
from utils.logger import get_logger
from utils.metrics import METRIC_REGISTRY
from tqdm.auto import tqdm

def get_dataloaders(config):
    dataset_fn = DATASET_REGISTRY[config['dataset']['name']]
    dataset = dataset_fn(cache_dir=config['dataset']['path'])
    collate_fn = get_collate_fn(config['dataset']['name'])
    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        dataset['validation'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset['test'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader, test_dataloader

def run_metrics(predictions, targets, model, inputs, device, metrics, results=None):
    if results is None:
        results = {metric: 0.0 for metric in metrics}
    for metric in metrics:
        results[metric] += METRIC_REGISTRY[metric](predictions=predictions, targets=targets, model=model, inputs=inputs, device=device).item()
    return results
    
def eval_model(model, dataloader, loss_fns, device, metrics):
    model.eval()
    results = {metric: 0.0 for metric in metrics}
    results['loss'] = 0.0
    for loss in loss_fns:
        results[loss['name']] = 0.0
    
    for inputs, targets, metadata in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss_values = torch.tensor([loss['fn'](predictions, targets) * loss['weight'] for loss in loss_fns])
            results['loss'] += torch.sum(loss_values).item()
            results = run_metrics(predictions, targets, model, inputs, device, metrics, results)
            for i, loss in enumerate(loss_fns):
                results[loss['name']] += loss_values[i].item()

    for metric in results:
        results[metric] /= len(dataloader)
    return results

def test_model(config):
    # Load environment variables
    load_dotenv(dotenv_path='noisescaling/.env')

    # Set seed for reproducibility
    set_seed(config['training']['seed'])

    # Create logger
    logger = get_logger(config['training']['experiment_name'])

    # Create model checkpoint directory
    base_model_dir = os.getenv('MODELS_DIR', 'models_cache')
    model_dir = os.path.join(base_model_dir, config['training']['experiment_name'])
    os.makedirs(model_dir, exist_ok=True)

    # Initialize wandb
    run = wandb.init(
        project=os.getenv('WANDB_PROJECT'),
        entity=os.getenv('WANDB_ENTITY'),
        name=config['training']['experiment_name'],
        id=config['training'].get('run_id', None),
        resume="must" if config['training'].get('run_id', None) else None,
        config=config
    )

    # Get the model from the registry
    model_class = MODEL_REGISTRY[config['model']['name']]
    model = model_class()
    logger.info(f"Model Architecture:\n{model}")
    # Load pretrained model, if provided.
    if config['model']['pretrained_path']:
        model.load_state_dict(torch.load(config['model']['pretrained_path']), strict=False)
    run.watch(model)

    model.eval()
    
    num_trainable_params = count_trainable_parameters(model)
    logger.info(f"Number of trainable parameters: {num_trainable_params}")
    run.config.update({"num_trainable_params": num_trainable_params})

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the dataloaders
    _, _, test_dataloader = get_dataloaders(config)
    
    # Get the loss function from the registry
    loss_fns = [{'name': loss['name'], 'fn': LOSS_REGISTRY[loss['name']], 'weight': loss['weight']} for loss in config['model']['losses']]

    # Evaluate the final model on the test set
    eval_results = eval_model(model, test_dataloader, loss_fns, device, config['training']['validation_metrics'])
    logger.info(f"Test results: {eval_results}")
    eval_results = {f"test/{k}": v for k, v in eval_results.items()}
    run.log(eval_results)

    # Finish the wandb run
    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    test_model(config)
