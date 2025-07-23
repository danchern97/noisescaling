import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml

# Import wandb
import wandb
from models.cnn import SudokuCNN

# 2. Training function
def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # print every 100 mini-batches
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {running_loss / 100:.6f}"
            )
            # Log training loss to wandb
            wandb.log({"train_loss": running_loss / 100})
            running_loss = 0.0


# 3. Validation function
def validate(model, device, val_loader, criterion, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100.0 * correct / len(val_loader.dataset)
    print(
        f"\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)}"
        f" ({accuracy:.2f}%)\n"
    )
    # Log validation metrics to wandb
    wandb.log({"val_loss": val_loss, "val_accuracy": accuracy, "epoch": epoch})
    return accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="directory to save the model (default: models)",
    )
    # Add wandb arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="pytorch-cifar10-example",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file.",
    )

    args = parser.parse_args()

    # Load config from YAML file if provided
    if args.config:
        with open(args.config, "r") as f:
            config_from_file = yaml.safe_load(f)
    else:
        config_from_file = {}

    # Merge argparse arguments into the config dictionary
    # CLI arguments take precedence over the YAML file
    config_dict = vars(args)
    for key, value in config_from_file.items():
        if key not in config_dict or config_dict[key] is None or config_dict[key] == parser.get_default(key):
             config_dict[key] = value


    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config_dict)

    # Use wandb.config for hyperparameters for consistency
    config = wandb.config

    use_cuda = not config.no_cuda and torch.cuda.is_available()
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loading
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model, Optimizer, Loss
    model = SudokuCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    # Watch model with wandb
    wandb.watch(model, criterion, log="all", log_freq=100)

    best_accuracy = 0.0

    if not os.path.exists(config.model_dir) and config.save_model:
        os.makedirs(config.model_dir)

    # Training loop
    for epoch in range(1, config.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, criterion)
        accuracy = validate(model, device, val_loader, criterion, epoch)

        # Save the model with the best validation accuracy
        if config.save_model and accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save model as a wandb artifact
            model_artifact = wandb.Artifact(
                f"cifar10-cnn-{wandb.run.id}",
                type="model",
                description="Best performing CNN model for CIFAR-10.",
                metadata={"epoch": epoch, "accuracy": best_accuracy, "lr": config.lr},
            )
            model_path = os.path.join(wandb.run.dir, "cifar10_cnn_best.pth")
            torch.save(model.state_dict(), model_path)
            model_artifact.add_file(model_path)
            wandb.log_artifact(model_artifact, aliases=["best"])
            print(f"Model artifact saved with accuracy {best_accuracy:.2f}%")

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
