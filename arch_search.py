# Architecture search
# Given the list of scalers and aggregators, along with the list of hyperparameters for each of them, search for the best architecture.
# ToDo:
# - Create a list of scalers and aggregators with corresponding hyperparameters.
# - For each combination, create a sweep yaml and initialize the sweep.
# - Run the sweep with agent, possibly in parallel.
import argparse
import yaml

def arch_search(config):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Architecture search.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()