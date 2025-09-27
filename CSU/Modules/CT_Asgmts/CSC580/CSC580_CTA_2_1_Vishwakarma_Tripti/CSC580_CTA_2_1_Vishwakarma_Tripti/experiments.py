#!/usr/bin/env python3
"""
Runs a sweep of hyperparameters for the MNIST MLP and collects results.
Saves results to outputs/experiments_results.json and simple plots.
"""
import os
import json
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from mnist_mlp_tf1 import run_experiment


def sweep(hidden_units_list, learning_rates, batch_sizes, layers_list, epochs=20, seed=42, work_dir='outputs'):
    os.makedirs(work_dir, exist_ok=True)
    results = []

    for hu, lr, bs, lyr in product(hidden_units_list, learning_rates, batch_sizes, layers_list):
        print(f"\n=== Running: hidden_units={hu}, lr={lr}, batch_size={bs}, layers={lyr} ===")
        res = run_experiment(hidden_units=hu,
                             learning_rate=lr,
                             batch_size=bs,
                             epochs=epochs,
                             layers=lyr,
                             seed=seed,
                             save_miscl=10,
                             work_dir=work_dir)
        results.append(res)

    with open(os.path.join(work_dir, 'experiments_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_accuracy_by_param(results, param_name, out_path):
    # Group by param and take best accuracy per value
    data = {}
    for r in results:
        key = r[param_name]
        data.setdefault(key, 0.0)
        data[key] = max(data[key], r['test_accuracy'])

    xs = sorted(data.keys())
    ys = [data[x] for x in xs]

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='o')
    plt.title(f'Best Accuracy vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    work_dir = 'outputs'
    results = sweep(hidden_units_list=[128, 512, 1024],
                    learning_rates=[0.1, 0.2, 0.5, 1.0],
                    batch_sizes=[32, 100, 256],
                    layers_list=[1, 2],
                    epochs=5,  # quick sweep; change to 20 for final best run
                    seed=42,
                    work_dir=work_dir)

    # Save simple plots for the report
    plot_accuracy_by_param(results, 'hidden_units', os.path.join(work_dir, 'acc_vs_hidden_units.png'))
    plot_accuracy_by_param(results, 'learning_rate', os.path.join(work_dir, 'acc_vs_learning_rate.png'))
    plot_accuracy_by_param(results, 'batch_size', os.path.join(work_dir, 'acc_vs_batch_size.png'))

    # Best configuration
    best = max(results, key=lambda r: r['test_accuracy'])
    with open(os.path.join(work_dir, 'best_config.json'), 'w') as f:
        json.dump(best, f, indent=2)
    print("\nBest configuration:", best)


if __name__ == '__main__':
    main()

