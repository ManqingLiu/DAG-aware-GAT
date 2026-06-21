import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import wandb
from typing import Dict

from src.models.dag_aware_transformer_loss import g_formula_loss_fun, ipw_loss_fun, aipw_loss_fun
from src.dataset import CausalDataset
from src.utils import predict_function, replace_column_values

# Import the pure DAG transformer
from src.models.pure_dag_transformer import \
    DAGTransformer


def train_pure_dag_transformer(
        data_name: str,
        estimator: str,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        val_data: pd.DataFrame,
        pseudo_ate_data: pd.DataFrame,
        sample_id: int,
        config: Dict,
        dag: Dict,
        random_seed: int = None,
        dag_attention_mask: bool = False,
        eval_metric: str = "nrmse"
) -> nn.Module:
    """
    Training function for Pure DAG Transformer

    Args:
        data_name: Name of the dataset
        estimator: Estimation method (g-formula, ipw, or aipw)
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        val_data: Validation data
        pseudo_ate_data: Pseudo ATE data
        sample_id: Sample ID
        config: Configuration dictionary
        dag: DAG structure
        random_seed: Random seed
        dag_attention_mask: Whether to use DAG-based attention masking
        eval_metric: Evaluation metric to use - "nrmse", "mae", or "both"

    Returns:
        Trained model, predictions, validation metrics
    """
    train_config = config[estimator]["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        weight_decay=train_config["l2_penalty"],
        lr=train_config["learning_rate"],
    )

    wandb.init(project="DAG Transformer", entity="mliu7", config=config)

    for epoch in range(train_config["n_epochs"]):
        model.train()
        total_train_loss = 0.0

        for batch_ix, (batch_raw, batch_binned) in enumerate(train_dataloader):
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch_raw.items()}
            outputs = model(batch, mask=dag_attention_mask, estimator=estimator)

            if estimator == "g-formula":
                y = batch_raw['y'].to(device).float()
                y_ = torch.squeeze(outputs['y']).to(device).float()
                batch_loss, batch_items = g_formula_loss_fun(y_, y)
            elif estimator == "ipw":
                t = batch_raw['t'].to(device).float()
                e = outputs['t'].to(device).squeeze().float()
                batch_loss, batch_items = ipw_loss_fun(e, t)
            else:  # aipw
                y = batch_raw['y'].to(device).float()
                y_ = torch.squeeze(outputs['y']).to(device).float()
                t = batch_raw['t'].to(device)
                e = outputs['t'].to(device).squeeze()
                batch_loss, batch_items = aipw_loss_fun(y_, y, e, t)

            batch_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_train_loss += batch_loss.item()

        # Log average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_raw_val, batch_binned_val in val_dataloader:
                batch_val = {k: v.to(device) for k, v in batch_raw_val.items()}
                outputs_val = model(batch_val, mask=dag_attention_mask, estimator=estimator)

                if estimator == "g-formula":
                    y = batch_val['y'].to(device).float()
                    y_ = torch.squeeze(outputs_val['y']).to(device).float()
                    val_batch_loss, val_batch_items = g_formula_loss_fun(y_, y)
                elif estimator == "ipw":
                    t = batch_val['t'].to(device).float()
                    e = outputs_val['t'].to(device).squeeze().float()
                    val_batch_loss, val_batch_items = ipw_loss_fun(e, t)
                else:  # aipw
                    y = batch_val['y'].to(device).float()
                    y_ = torch.squeeze(outputs_val['y']).to(device).float()
                    t = batch_val['t'].to(device).float()
                    e = outputs_val['t'].to(device).squeeze().float()
                    val_batch_loss, val_batch_items = aipw_loss_fun(y_, y, e, t)

                val_loss += val_batch_loss.item()

            val_loss_avg = val_loss / len(val_dataloader)
            wandb.log({"Validation Loss": val_loss_avg, "Epoch": epoch})

        # Get predictions and compute metrics
        predictions, metrics_val = predict_pure_dag_transformer(
            model=model,
            data_name=data_name,
            data=val_data,
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

        # Extract metrics based on chosen metric type
        if estimator == "g-formula":
            test_ate = metrics_val["Test: predicted ATE for standardization"]
            test_nrmse = metrics_val["Test: NRMSE for standardization"]
            test_mae = metrics_val["Test: MAE for standardization"]
            test_mae_std = metrics_val["Test: MAE std for standardization"]
        elif estimator == "ipw":
            test_ate = metrics_val["Test: predicted ATE for IPW"]
            test_nrmse = metrics_val["Test: NRMSE for IPW"]
            test_mae = metrics_val["Test: MAE for IPW"]
            test_mae_std = metrics_val["Test: MAE std for IPW"]
        else:  # aipw
            test_ate = metrics_val["Test: predicted ATE for AIPW"]
            test_nrmse = metrics_val["Test: NRMSE for AIPW"]
            test_mae = metrics_val["Test: MAE for AIPW"]
            test_mae_std = metrics_val["Test: MAE std for AIPW"]

        # Log average training loss and test metric(s) for the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Print metrics
        if eval_metric == "both":
            print(
                f"Epoch {epoch + 1}/{train_config['n_epochs']}, Test ATE: {test_ate:.4f}, "
                f"Test NRMSE: {test_nrmse:.4f}, Test MAE: {test_mae:.4f} ± {test_mae_std:.4f}")
        elif eval_metric == "mae":
            print(
                f"Epoch {epoch + 1}/{train_config['n_epochs']}, Test ATE: {test_ate:.4f}, "
                f"Test MAE: {test_mae:.4f} ± {test_mae_std:.4f}")
        else:  # nrmse
            print(
                f"Epoch {epoch + 1}/{train_config['n_epochs']}, Test ATE: {test_ate:.4f}, "
                f"Test NRMSE: {test_nrmse:.4f}")

        # Log to wandb
        log_dict = {
            "Train Loss": avg_train_loss,
            "Test ATE": test_ate,
            "Epoch": epoch
        }
        if eval_metric in ["nrmse", "both"]:
            log_dict["Test NRMSE"] = test_nrmse
        if eval_metric in ["mae", "both"]:
            log_dict["Test MAE"] = test_mae
            log_dict["Test MAE std"] = test_mae_std
        wandb.log(log_dict)

    return model, predictions, metrics_val


def predict_pure_dag_transformer(
        model,
        data_name,
        data,
        pseudo_ate_data,
        dag,
        train_config: Dict,
        random_seed: int,
        sample_id: int,
        prefix: str = "Test",
        estimator: str = "ipw",
        dag_attention_mask: bool = False,
        eval_metric: str = "nrmse"
):
    """
    Prediction function for Pure DAG Transformer

    Args:
        model: Trained model
        data_name: Name of the dataset
        data: Data to predict on
        pseudo_ate_data: Pseudo ATE data
        dag: DAG structure
        train_config: Training configuration
        random_seed: Random seed
        sample_id: Sample ID
        prefix: Prefix for metrics
        estimator: Estimation method
        dag_attention_mask: Whether to use DAG-based attention masking
        eval_metric: Evaluation metric to use - "nrmse", "mae", or "both"

    Returns:
        Predictions, metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = train_config["batch_size"]
    model = model.to(device)

    data_nodes = data[dag['nodes']]
    dataset = CausalDataset(data_nodes, dag, random_seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    data_A0 = replace_column_values(data_nodes, "t", 0)
    dataset_A0 = CausalDataset(data_A0, dag, random_seed)
    dataloader_A0 = DataLoader(
        dataset_A0,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    data_A1 = replace_column_values(data_nodes, "t", 1)
    dataset_A1 = CausalDataset(data_A1, dag, random_seed)
    dataloader_A1 = DataLoader(
        dataset_A1,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Generate predictions based on the estimator type
    if estimator == "g-formula":
        predictions_y0 = predict_outputs(model, dataloader_A0, estimator, dag_attention_mask)['y']
        predictions_y0 = pd.DataFrame(predictions_y0, columns=['pred_y_A0'])

        predictions_y1 = predict_outputs(model, dataloader_A1, estimator, dag_attention_mask)['y']
        predictions_y1 = pd.DataFrame(predictions_y1, columns=['pred_y_A1'])

        predictions = pd.concat(
            [data, predictions_y0["pred_y_A0"], predictions_y1["pred_y_A1"]],
            axis=1,
        )
    elif estimator == 'ipw':
        predictions_t = predict_outputs(model, dataloader, estimator, dag_attention_mask)['t']
        predictions_t = pd.DataFrame(predictions_t, columns=['t_prob'])

        predictions = pd.concat(
            [data, predictions_t["t_prob"]],
            axis=1,
        )
    else:  # aipw
        predictions_y0 = predict_outputs(model, dataloader_A0, estimator, dag_attention_mask)['y']
        predictions_y0 = pd.DataFrame(predictions_y0, columns=['pred_y_A0'])

        predictions_y1 = predict_outputs(model, dataloader_A1, estimator, dag_attention_mask)['y']
        predictions_y1 = pd.DataFrame(predictions_y1, columns=['pred_y_A1'])

        predictions_t = predict_outputs(model, dataloader, estimator, dag_attention_mask)['t']
        predictions_t = pd.DataFrame(predictions_t, columns=['t_prob'])

        predictions = pd.concat(
            [data, predictions_y0["pred_y_A0"], predictions_y1["pred_y_A1"], predictions_t["t_prob"]],
            axis=1,
        )

    # Calculate metrics based on the dataset
    metrics = None

    # Import the appropriate metrics calculation function based on the data_name
    if data_name == "lalonde_cps" or data_name == "lalonde_psid":
        if prefix == "Val":
            from src.train.lalonde.train_metrics import calculate_val_metrics
            metrics = calculate_val_metrics(
                predictions,
                pseudo_ate_data,
                sample_id,
                prefix=prefix,
                estimator=estimator,
                eval_metric=eval_metric
            )
        else:  # "Test"
            from src.evaluate.lalonde.evaluate_metrics import calculate_test_metrics
            metrics = calculate_test_metrics(
                predictions,
                prefix=prefix,
                estimator=estimator,
                eval_metric=eval_metric
            )
    elif data_name == "acic":
        if prefix == "Val":
            from src.train.acic.train_metrics import calculate_val_metrics_acic
            metrics = calculate_val_metrics_acic(
                predictions,
                pseudo_ate_data,
                prefix=prefix,
                estimator=estimator,
                sample_id=sample_id,
                eval_metric=eval_metric
            )
        else:  # "Test"
            from src.evaluate.acic.evaluate_metrics import calculate_test_metrics_acic
            metrics = calculate_test_metrics_acic(
                predictions,
                data['mu1'] - data['mu0'],
                prefix=prefix,
                estimator=estimator,
                eval_metric=eval_metric
            )

    return predictions, metrics


def predict_outputs(model, dataloader, estimator, dag_attention_mask: bool = False):
    """
    Helper function to generate model predictions

    Args:
        model: Trained model
        dataloader: Data loader
        estimator: Estimation method
        dag_attention_mask: Whether to use DAG-based attention masking

    Returns:
        Dictionary of predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_outputs = {}

    with torch.no_grad():
        for batch_raw, _ in dataloader:
            batch = {k: v.to(device) for k, v in batch_raw.items()}
            outputs = model(batch, mask=dag_attention_mask, estimator=estimator)

            # Initialize output arrays if not done yet
            if not all_outputs:
                for key in outputs:
                    all_outputs[key] = []

            # Append batch outputs
            for key, value in outputs.items():
                all_outputs[key].append(value.cpu().numpy())

    # Concatenate batches and squeeze dimensions if needed
    for key in all_outputs:
        all_outputs[key] = np.concatenate(all_outputs[key], axis=0)
        if all_outputs[key].shape[-1] == 1:
            all_outputs[key] = np.squeeze(all_outputs[key], axis=-1)

    return all_outputs
