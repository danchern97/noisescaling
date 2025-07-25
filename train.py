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

    # Set up the scaler and aggregator
    scaler, aggregator = None, None
    if config['model'].get('scaler', None):
        scaler = get_model_by_name(config['model']['scaler']['name'], **config['model']['scaler']['args'])
    if config['model'].get('aggregator', None):
        aggregator = get_model_by_name(config['model']['aggregator']['name'], **config['model']['aggregator']['args'])

    # Get the model from the registry
    model = get_model_by_name(config['model']['name'], scaler=scaler, aggregator=aggregator)
    logger.info(f"Model Architecture:\n{model}")
    # Load pretrained weights, if provided.
    if config['model']['pretrained_path']:
        model.load_state_dict(torch.load(config['model']['pretrained_path']), strict=False)
    
    run.watch(model)
    
    num_trainable_params = count_trainable_parameters(model)
    logger.info(f"Number of trainable parameters: {num_trainable_params}")
    run.config.update({"num_trainable_params": num_trainable_params})

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the dataloaders
    dataloaders = get_dataloaders(config)
    
    # Set up the optimizer
    if config['training']['scaler_regime'] == 'full':
        parameters = model.parameters()
    elif config['training']['scaler_regime'] == 'partial':
        parameters = list(model.scaler.parameters()) + list(model.aggregator.parameters())
    else:
        raise ValueError(f"Invalid scaler regime: {config['training']['scaler_regime']}. Supported regimes are 'full' and 'partial'.")
    optimizer = AdamW(parameters, lr=config['training']['learning_rate'])
    
    # Get the loss function from the registry
    loss_fns = [{'name': loss['name'], 'fn': LOSS_REGISTRY[loss['name']], 'weight': loss['weight']} for loss in config['model']['losses']]
    model.compile()
    model.train()
    step = 0
    for epoch in range(config['training']['epochs']):
        for _, (inputs, targets, _) in enumerate(tqdm(dataloaders['train'], desc='Training')):
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            # Multiple losses are summed up with corresponding coefficients
            loss_values = torch.stack([loss['fn'](predictions, targets) * loss['weight'] for loss in loss_fns], dim=0)
            loss = torch.sum(loss_values)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            log_dict = run_metrics(predictions, targets, model, inputs, device, config['training']['training_metrics'])
            for i, loss_fn in enumerate(loss_fns):
                log_dict[loss_fn['name']] = loss_values[i].item()
            log_dict['loss'] = loss.item()
            log_dict = {f"train/{k}": v for k, v in log_dict.items()}
            run.log(log_dict)
            step += 1

            if step % config['training']['eval_interval'] == 0 and 'validation' in dataloaders:
                eval_results = eval_model(model, dataloaders['validation'], loss_fns, device, config['training']['validation_metrics'])
                log_msg = f"Validation results at epoch {epoch}, step {step}: "
                log_msg += ", ".join([f"{k}: {v:.4f}" for k, v in eval_results.items()])
                logger.info(log_msg)
                log_dict = {f"val/{k}": v for k, v in eval_results.items()}
                run.log(log_dict)
                
                # Save model checkpoint
                model_path = os.path.join(model_dir, f"model_{step}.pt")
                torch.save(model.state_dict(), model_path)
                run.save(model_path)

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
