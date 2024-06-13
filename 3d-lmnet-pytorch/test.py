import os
import yaml
import argparse
from inference import inference_2d_to_3d, infer_3d

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def test():
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--experiment', type=str, required=True, help='Type of the experiment')
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)

    if args.experiment == "2d_to_3d":
        inference_2d_to_3d.main(config)
    elif args.experiment == "3d":
        infer_3d.main(config)
    else:
        raise ValueError(f"Experiment {args.experiment} not valid.")

if __name__ == "__main__":
    test()