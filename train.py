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


def run_metrics(predictions, targets, model, inputs, device, metrics, results=None):
    if results is None:
        results = {metric: 0.0 for metric in metrics}
    for metric in metrics:
        results[metric] += METRIC_REGISTRY[metric](predictions=predictions, targets=targets, model=model, inputs=inputs, device=device).item()
    return results
    
def eval_model(model, dataloader, loss_fn, device, metrics):
    model.eval()
    results = {metric: 0.0 for metric in metrics}
    results['loss'] = 0.0
    
    for inputs, targets, metadata in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            results['loss'] += loss_fn(predictions=predictions, targets=targets, model=model, inputs=inputs, device=device).item()
            results = run_metrics(predictions, targets, model, inputs, device, metrics, results)

    for metric in results:
        results[metric] /= len(dataloader)
    return results

def train_model(config):
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
        config=config
    )

    # Get the model from the registry
    model_class = MODEL_REGISTRY[config['model']['name']]
    model = model_class()
    logger.info(f"Model Architecture:\n{model}")
    run.watch(model)
    
    num_trainable_params = count_trainable_parameters(model)
    logger.info(f"Number of trainable parameters: {num_trainable_params}")
    run.config.update({"num_trainable_params": num_trainable_params})

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the dataset from the registry
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
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Get the loss function from the registry
    loss_fn = LOSS_REGISTRY[config['model']['loss']]

    model.train()
    for epoch in range(config['training']['epochs']):
        for i, (inputs, targets, metadata) in enumerate(tqdm(train_dataloader, desc='Training')):
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            log_dict = run_metrics(predictions, targets, model, inputs, device, config['training']['training_metrics'])
            log_dict['loss'] = loss.item()
            log_dict = {f"train/{k}": v for k, v in log_dict.items()}
            run.log(log_dict)

            if i % config['training']['eval_interval'] == 0:
                eval_results = eval_model(model, val_dataloader, loss_fn, device, config['training']['validation_metrics'])
                log_msg = f"Validation results at epoch {epoch}, step {i}: "
                log_msg += ", ".join([f"{k}: {v:.4f}" for k, v in eval_results.items()])
                logger.info(log_msg)
                log_dict = {f"val/{k}": v for k, v in eval_results.items()}
                run.log(log_dict)
                
                # Save model checkpoint
                model_path = os.path.join(model_dir, f"model_{epoch}_{i}.pt")
                torch.save(model.state_dict(), model_path)
                run.save(model_path)

                model.train()
    
    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_model(config)
