"""Run the dataset simulator; stages are defined in the simulator package.

From the repository root:
    python src/simulate.py --customers 10 --out data/simulator_demo

Then calculate the features:
    python src/features.py --data data/simulator_demo

The original single-file version is retained locally in archived/simulate.py (not included in Git).
Read simulator/run.py for the generation sequence.
"""
import argparse
import time

import yaml

from simulator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/generator.yaml")
    parser.add_argument("--out", default="data/simulator_demo")
    parser.add_argument("--customers", type=int, help="Override the configured customer count")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    if args.customers:
        config["n_customers"] = args.customers
    if args.seed is not None:
        config["seed"] = args.seed

    started = time.time()
    generate_dataset(config, args.out)
    print(f"Simulated {config['n_customers']} customers in {time.time()-started:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
