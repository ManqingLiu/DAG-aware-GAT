from typing import Dict
import numpy as np
import pandas as pd
from src.utils import rmse


# pseudo_ate is estimated using covariance balancing PS estimator on validation set
def std_rmse(mu0: np.array, mu1: np.array, pseudo_ite: np.array) -> float:
    """Plug-in estimator, equivalent to standardization."""
    ite_pred = mu1 - mu0
    return rmse(pseudo_ite, ite_pred)


def ipw_rmse(y: np.ndarray, t: np.ndarray, ps: np.ndarray, pseudo_ite: np.array) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    return rmse(pseudo_ite, ite_pred)

def cfcv_rmse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray, pseudo_ite: np.array) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    ite_pred = (t * (y - mu1) / ps) - ((1 - t) * (y - mu0) / (1 - ps)) + (mu1 - mu0)
    return rmse(pseudo_ite, ite_pred)

def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2)

def calculate_val_metrics_acic(
    predictions: np.array,
    pseudo_ite: pd.DataFrame,
    prefix: str,
    estimator: str,
    sample_id: int,
    eval_metric: str = "nrmse"
) -> Dict[str, float]:
    predictions['ite'] = predictions['mu1']-predictions['mu0']
    pseudo_ite_value = pseudo_ite.iloc[sample_id]["ate"]
    if estimator == "g-formula":
        std_rmse_ = std_rmse(predictions['pred_y_A0'], predictions['pred_y_A1'], pseudo_ite_value)
        return {f"{prefix}: RMSE for standardization": std_rmse_}
    elif estimator == "ipw":
        ipw_rmse_ = ipw_rmse(predictions['y'], predictions['t'], predictions['t_prob'], pseudo_ite_value)
        return {f"{prefix}: RMSE for IPW": ipw_rmse_}
    else:
        cfcv_rmse_ = cfcv_rmse(predictions['y'], predictions['t'], predictions['mu0'],
                           predictions['mu1'], predictions['t_prob'], pseudo_ite_value)
        return {f"{prefix}: RMSE for CFCV": cfcv_rmse_}