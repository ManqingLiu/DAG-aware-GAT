from typing import Dict
import numpy as np
import pandas as pd

from src.utils import rmse

# pseudo_ate is estimated using counterfactual cross-validation via causal forest (AIPW)
def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2))

def std_nrmse(mu0: np.array, mu1: np.array, pseudo_ate: float) -> float:
    """Plug-in estimator, equivalent to standardization."""
    ite_pred = mu1 - mu0
    ate_pred = np.mean(ite_pred)
    return nrmse(pseudo_ate, ate_pred)

def std_mae(mu0: np.array, mu1: np.array, pseudo_ate: float) -> tuple:
    """Plug-in estimator with MAE metric.
    For validation, we only have pseudo ATE (scalar), so MAE = |predicted_ATE - pseudo_ATE| and std = 0."""
    ite_pred = mu1 - mu0
    ate_pred = np.mean(ite_pred)
    mae = np.abs(ate_pred - pseudo_ate)
    mae_std = 0.0
    return mae, mae_std

def ipw_nrmse(y: np.ndarray, t: np.ndarray, ps: np.ndarray, pseudo_ate: float) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    return nrmse(pseudo_ate, ate_pred)

def ipw_mae(y: np.ndarray, t: np.ndarray, ps: np.ndarray, pseudo_ate: float) -> tuple:
    """MAE with inverse propensity weighting.
    For validation, we only have pseudo ATE (scalar), so MAE = |predicted_ATE - pseudo_ATE| and std = 0."""
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    mae = np.abs(ate_pred - pseudo_ate)
    mae_std = 0.0
    return mae, mae_std


def aipw_nrmse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray,  pseudo_ate: float) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    ite_pred = (t * (y - mu1) / ps) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0)
    ate_pred = np.mean(ite_pred)
    return nrmse(pseudo_ate, ate_pred)

def aipw_mae(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray, pseudo_ate: float) -> tuple:
    """MAE with Counterfactual Cross Validation, equivalent to doubly robust estimator.
    For validation, we only have pseudo ATE (scalar), so MAE = |predicted_ATE - pseudo_ATE| and std = 0."""
    ite_pred = (t * (y - mu1) / ps) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0)
    ate_pred = np.mean(ite_pred)
    mae = np.abs(ate_pred - pseudo_ate)
    mae_std = 0.0
    return mae, mae_std

def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2)

def mae_with_std(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Mean Absolute Error with standard deviation."""
    abs_errors = np.abs(y_true - y_pred)
    mae = np.mean(abs_errors)
    mae_std = np.std(abs_errors)
    return mae, mae_std

def calculate_val_metrics(
    predictions: np.array,
    pseudo_ate: pd.DataFrame,
    sample_id: int,
    estimator: str,
    prefix: str,
    eval_metric: str = "nrmse"
) -> Dict[str, float]:
    pseudo_ate_value = pseudo_ate.iloc[sample_id]["rmse_ate"]
    if estimator == "g-formula":
        std_nrmse_ = std_nrmse(predictions['pred_y_A0'], predictions['pred_y_A1'], pseudo_ate_value)
        std_mae_, std_mae_std = std_mae(predictions['pred_y_A0'], predictions['pred_y_A1'], pseudo_ate_value)
        return {
                f"{prefix}: NRMSE for standardization": std_nrmse_,
                f"{prefix}: MAE for standardization": std_mae_,
                f"{prefix}: MAE std for standardization": std_mae_std}
    elif estimator == "ipw":
        ipw_nrmse_ = ipw_nrmse(predictions['y'], predictions['t'], predictions['t_prob'], pseudo_ate_value)
        ipw_mae_, ipw_mae_std = ipw_mae(predictions['y'], predictions['t'], predictions['t_prob'], pseudo_ate_value)
        return {
                f"{prefix}: NRMSE for IPW": ipw_nrmse_,
                f"{prefix}: MAE for IPW": ipw_mae_,
                f"{prefix}: MAE std for IPW": ipw_mae_std}
    else:
        aipw_nrmse_ = aipw_nrmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                                           predictions['pred_y_A1'], predictions['t_prob'], pseudo_ate_value)
        aipw_mae_, aipw_mae_std = aipw_mae(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                                           predictions['pred_y_A1'], predictions['t_prob'], pseudo_ate_value)
        return {
                f"{prefix}: NRMSE for AIPW": aipw_nrmse_,
                f"{prefix}: MAE for AIPW": aipw_mae_,
                f"{prefix}: MAE std for AIPW": aipw_mae_std}