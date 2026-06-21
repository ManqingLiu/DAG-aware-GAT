from argparse import ArgumentParser
import json
from typing import Dict
import time

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.utils_data import data_preprocess

# Import the Pure DAG Transformer implementation
from src.models.pure_dag_transformer import DAGTransformer
from src.models.dag_transformer_integration import train_pure_dag_transformer, predict_pure_dag_transformer

# Import the GAT Baseline implementation
from src.models.gat_baseline import GAT
from src.models.gat_integration import train_gat_baseline, predict_gat_baseline


def experiment(data_name: str,
               estimator: str,
               config: Dict,
               dag: Dict,
               train_dataloader: DataLoader,
               val_dataloader: DataLoader,
               val_data: pd.DataFrame,
               pseudo_ate_data: pd.DataFrame,
               random_seed=False,
               sample_id: int = None,
               test_data: pd.DataFrame = None,
               model_type: str = "dag_transformer",
               use_layernorm: bool = False,
               dag_attention_mask: bool = False,
               eval_metric: str = "nrmse"
               ):
    """
    Run experiment with specified model type.
    
    Args:
        data_name: Name of the dataset
        estimator: Estimation method (g-formula, ipw, or aipw)
        config: Configuration dictionary
        dag: DAG structure
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        val_data: Validation data
        pseudo_ate_data: Pseudo ATE data
        random_seed: Whether to set random seed
        sample_id: Sample ID
        test_data: Test data
        model_type: Model type - "dag_transformer" (default) or "gat" (baseline)
        use_layernorm: Whether to use layer normalization in transformer
        dag_attention_mask: Whether to use DAG-based attention masking
        eval_metric: Evaluation metric to use - "nrmse" (default), "mae", or "both"
        
    Returns:
        Trained model, predictions, and metrics
    """
    model_config = config[estimator]["model"]
    train_config = config[estimator]["training"]
    if random_seed:
        torch.manual_seed(config["random_seed"])

    if model_type == "gat":
        # Create GAT Baseline model (no DAG attention mask)
        model = GAT(
            dag=dag,
            network_width=model_config["network_width"],
            embedding_dim=model_config["embedding_dim"],
            feedforward_dim=model_config["feedforward_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            dropout_rate=model_config["dropout_rate"],
            input_layer_depth=model_config["input_layer_depth"],
            encoder_weight=model_config["encoder_weight"],
            activation=model_config.get("activation", "relu"),
            use_layernorm=use_layernorm,
            name=f"gat_baseline_{estimator}"
        )
        # Train the GAT model
        model, _, _ = train_gat_baseline(
            data_name=data_name,
            estimator=estimator,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            val_data=val_data,
            pseudo_ate_data=pseudo_ate_data,
            sample_id=sample_id,
            config=config,
            dag=dag,
            random_seed=random_seed,
            dag_attention_mask=dag_attention_mask,
            eval_metric=eval_metric
        )
        # Generate predictions on test data
        predictions_test, metrics_test = predict_gat_baseline(
            model=model,
            data_name=data_name,
            data=test_data,
            pseudo_ate_data=pseudo_ate_data,
            dag=dag,
            train_config=train_config,
            random_seed=random_seed,
            sample_id=sample_id,
            prefix="Test",
            estimator=estimator,
            dag_attention_mask=dag_attention_mask,
            eval_metric=eval_metric
        )
    else:
        # Create Pure DAG Transformer model (default)
        model = DAGTransformer(
            dag=dag,
            network_width=model_config["network_width"],
            embedding_dim=model_config["embedding_dim"],
            feedforward_dim=model_config["feedforward_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            dropout_rate=model_config["dropout_rate"],
            input_layer_depth=model_config["input_layer_depth"],
            encoder_weight=model_config["encoder_weight"],
            activation=model_config.get("activation", "relu"),
            use_layernorm=use_layernorm,
            name=f"dag_transformer_{estimator}"
        )
        # Train the model
        model, _, _ = train_pure_dag_transformer(
            data_name=data_name,
            estimator=estimator,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            val_data=val_data,
            pseudo_ate_data=pseudo_ate_data,
            sample_id=sample_id,
            config=config,
            dag=dag,
            random_seed=random_seed,
            dag_attention_mask=dag_attention_mask,
            eval_metric=eval_metric
        )
        # Generate predictions on test data
        predictions_test, metrics_test = predict_pure_dag_transformer(
            model=model,
            data_name=data_name,
            data=test_data,
            pseudo_ate_data=pseudo_ate_data,
            dag=dag,
            train_config=train_config,
            random_seed=random_seed,
            sample_id=sample_id,
            prefix="Test",
            estimator=estimator,
            dag_attention_mask=dag_attention_mask,
            eval_metric=eval_metric
        )

    return model, predictions_test, metrics_test


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--estimator", type=str, required=True)
    parser.add_argument("--dag", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--sample_id", type=int, required=False, default=None)
    parser.add_argument("--model_type", type=str, required=False, default="dag_transformer",
                        choices=["dag_transformer", "gat"],
                        help="Model type: 'dag_transformer' (default) or 'gat' (baseline without DAG)")
    parser.add_argument("--use_layernorm", action="store_true",
                        help="Use layer normalization in transformer layers")
    parser.add_argument("--dag_attention_mask", action="store_true",
                        help="Use DAG-based attention masking")
    parser.add_argument("--eval_metric", type=str, required=False, default="nrmse",
                        choices=["nrmse", "mae", "both"],
                        help="Evaluation metric: 'nrmse' (default), 'mae' (mean absolute error with std dev), or 'both'")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    filepaths = config["filepaths"]
    config_train = config[args.estimator]["training"]

    with open(filepaths[args.dag]) as f:
        dag = json.load(f)

    (train_data, train_dataloader, val_data, val_dataloader, test_data, test_dataloader) = data_preprocess(
        args.estimator, config, filepaths, dag
    )

    start_time = time.time()

    if "lalonde" in args.data_name:
        pseudo_ate_data = pd.read_csv(filepaths["pseudo_ate_file"])
    else:
        pseudo_ate_data = pd.read_csv(filepaths["pseudo_cate_file"])

    # Run experiment
    (model, predictions_test, metrics_test) = experiment(
        args.data_name,
        args.estimator,
        config,
        dag,
        train_dataloader,
        test_dataloader,
        test_data,
        pseudo_ate_data,
        sample_id=args.sample_id,
        random_seed=True,
        test_data=test_data,
        model_type=args.model_type,
        use_layernorm=args.use_layernorm,
        dag_attention_mask=args.dag_attention_mask,
        eval_metric=args.eval_metric)

    # Print metrics
    print(metrics_test)

    # Save predictions as CSV
    predictions_test.to_csv(filepaths[f"predictions_{args.estimator}"], index=False)

    # Calculate and print total wall time
    end_time = time.time()
    total_wall_time = end_time - start_time
    minutes, seconds = divmod(total_wall_time, 60)
    print(f"Total wall time used: {minutes:.0f} minutes and {seconds:.2f} seconds")