from typing import Dict, Any, Optional
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from src.utils_proximal import grid_search_dict
from src.models.NMMR.NMMR_experiments import NMMR_experiment


def experiments(configs: Dict[str, Any],
                dag: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int,
                sample_index: int,
                use_layernorm: bool = False,
                dag_attention_mask: bool = False):
    data_config = configs["data"]
    model_config = configs["model_transformer"]
    train_config = configs['training_transformer']
    num_nodes = len(dag['nodes'])
    dag['node_ids'] = dict(zip(dag['nodes'], range(num_nodes)))
    n_repeat: int = configs["n_repeat"]

    one_dump_dir = f"experiments/results/proximal/n_sample:{data_config['n_sample']}"

    print(f"Running sample {sample_index}")
    _, _, _, test_loss, _ = NMMR_experiment(
        configs, data_config, train_config, model_config, dag, one_dump_dir, random_seed=sample_index,
        use_layernorm=use_layernorm, dag_attention_mask=dag_attention_mask)
    if test_loss is not None:
        one_dump_dir = Path(one_dump_dir)
        np.savetxt(one_dump_dir.joinpath(f"result_{sample_index}.csv"), np.array([test_loss]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--sample_index', type=int, required=True)
    parser.add_argument('--use_layernorm', action='store_true',
                        help='Use layer normalization in transformer layers')
    parser.add_argument('--dag_attention_mask', action='store_true',
                        help='Use DAG-based attention masking')
    # Load the configurations from the JSON file
    args = parser.parse_args()

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    with open(args.config) as f:
        print(f'Loading config file from {args.config}')
        config = json.load(f)

    # Define the directory where the model will be saved
    dump_dir = Path(args.results_dir)

    # Define the number of CPUs to use
    num_cpus = 10  # replace with the number of CPUs you want to use

    sample_index = args.sample_index

    # Run the experiments
    experiments(config, dag, dump_dir, num_cpus, sample_index,
                use_layernorm=args.use_layernorm,
                dag_attention_mask=args.dag_attention_mask)