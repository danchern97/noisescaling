# noisescaling

Test-time scaling by sampling perturbations in the latent space.

## Setup

1.  **Clone the repository (optional):**
    ```bash
    git clone https://github.com/danchern97/noisescaling.git
    cd noisescaling
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your environment:**
    Create a `.env` file inside the `noisescaling` directory. This file will store your Weights & Biases configuration and other environment-specific settings.

    ```bash
    # noisescaling/.env
    WANDB_PROJECT="noisescaling"
    WANDB_ENTITY="your-wandb-entity" # Replace with your W&B username or team name
    LOGS_DIR="logs"
    ```

    You will also need to log in to Weights & Biases:
    ```bash
    wandb login
    ```

## Training a Model

To train a model, run the `train.py` script and specify the configuration file for your experiment.

```bash
python -m noisescaling.train --config noisescaling/config.yaml
```

All training parameters, model selection, and dataset choices are controlled through the configuration file.

## Running a Hyperparameter Sweep

To run a hyperparameter sweep, follow these steps:

1.  **Initialize the Sweep:**
    First, initialize the sweep using the `wandb sweep` command. This will create the sweep on Weights & Biases and print a `SWEEP_ID`.

    ```bash
    wandb sweep --project your-project-name sweep.yaml
    ```
    Replace `your-project-name` with your actual W&B project name.

2.  **Run the Agents:**
    Once you have the `SWEEP_ID`, you need to run the agents to perform the training runs. The `run_agents.sh` script is designed for this purpose.

    Open `run_agents.sh` and set the `SWEEP_ID` variable to the ID you received from the previous step. You can also configure the number of agents (`N_AGENTS`) and the number of runs per agent (`COUNT_PER_AGENT`).

    ```bash
    # run_agents.sh
    PROJECT="your-project-name"
    ENTITY="your-wandb-entity"
    SWEEP_ID="your-sweep-id" # <-- PASTE YOUR SWEEP ID HERE

    N_AGENTS=3
    COUNT_PER_AGENT=4
    # ...
    ```

    Then, execute the script to start the agents:
    ```bash
    ./run_agents.sh
    ```
    This will launch multiple agents in the background, each running a set number of experiments from the sweep. You can monitor the progress of the sweep from your Weights & Biases dashboard.

## Extending the Framework

This framework is designed to be easily extensible. Here’s how you can add your own components.

### Adding a New Model

1.  Create a new Python file in `noisescaling/models/` (e.g., `noisescaling/models/my_model.py`).
2.  In this file, define your model as a class that inherits from `torch.nn.Module`.
3.  Import the `register_model` and `register_loss` decorators from `noisescaling.models`.
4.  Decorate your model class with `@register_model("your-model-name")`.
5.  Define and decorate a corresponding loss function with `@register_loss("your-loss-name")`.
6.  Import your new model file in `noisescaling/models/__init__.py` (e.g., `from . import my_model`).
7.  Update your `config.yaml` to use your new model and loss.

**Example (`noisescaling/models/my_model.py`):**
```python
import torch.nn as nn
from . import register_model, register_loss

@register_model("MyAwesomeModel")
class MyAwesomeModel(nn.Module):
    # ... your model implementation ...
    pass

@register_loss("my_awesome_loss")
def get_my_awesome_loss(predictions, targets):
    # ... your loss calculation ...
    pass
```

### Adding a New Dataset

1.  Open the `noisescaling/data_processing/__init__.py` file.
2.  Create a function that loads and preprocesses your dataset. It should return a `datasets.DatasetDict`.
3.  Decorate this function with `@register_dataset("your-dataset-name")`.
4.  Create a custom `collate_fn` for your dataset.
5.  Decorate the collate function with `@register_collate_fn("your-dataset-name")`.

**Example (`noisescaling/data_processing/__init__.py`):**
```python
# ... existing imports ...
from . import register_dataset, register_collate_fn

@register_dataset("my_dataset")
def load_my_dataset(cache_dir, **kwargs):
    # ... load and process your dataset ...
    return dataset_dict

@register_collate_fn("my_dataset")
def collate_fn_my_dataset(batch):
    # ... your custom collate logic ...
    return inputs, targets, metadata
```

### Adding a New Metric

1.  Open the `noisescaling/utils/metrics.py` file.
2.  Create a function that takes `predictions` and `targets` as input and returns a scalar value (the metric score).
3.  Decorate your function with `@register_metric("your-metric-name")`.

**Example (`noisescaling/utils/metrics.py`):**
```python
from . import register_metric

@register_metric("my_metric")
def get_my_metric(predictions, targets, **kwargs):
    # ... calculate your metric ...
    return score
```
To use your new metric, add its registered name to the `metrics` list in your `config.yaml`.