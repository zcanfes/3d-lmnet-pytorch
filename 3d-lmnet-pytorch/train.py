import torch
import yaml
import argparse
import os
from data.shapenet import ShapeNet
from model.model_2d import ImageEncoder
from torchsummary import summary
from training import train_ae, train_2d_to_3d


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train():
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--experiment', type=str, required=True, help='Type of the experiment')
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)
    
    if args.experiment == "2d_to_3d":
        train_2d_to_3d.main(config)
    elif args.experiment == "3d":
        train_ae.main(config)
    else:
        raise ValueError(f"Experiment type {args.experiment} not valid.")


if __name__ == '__main__':
    train()