from typing import Dict
import numpy as np
from src.utils import rmse
from argparse import ArgumentParser
import pandas as pd

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2))

def mae_with_std(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Mean Absolute Error with standard deviation."""
    abs_errors = np.abs(y_true - y_pred)
    mae = np.mean(abs_errors)
    mae_std = np.std(abs_errors)
    return mae, mae_std

def std_nrmse(mu0: np.array, mu1: np.array, true_ite: np.array) -> float:
    """Plug-in estimator, equivalent to standardization."""
    ite_pred = mu1 - mu0
    ate_pred = np.mean(ite_pred)
    return ate_pred, nrmse(true_ite, ite_pred)

def std_mae(mu0: np.array, mu1: np.array, true_ite: np.array) -> tuple:
    """Plug-in estimator with MAE metric."""
    ite_pred = mu1 - mu0
    ate_pred = np.mean(ite_pred)
    mae, mae_std = mae_with_std(true_ite, ite_pred)
    return ate_pred, mae, mae_std


def ipw_nrmse(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ite: np.array) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    return ate_pred, nrmse(true_ite, ite_pred)

def ipw_mae(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ite: np.array) -> tuple:
    """MAE with inverse propensity weighting"""
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    mae, mae_std = mae_with_std(true_ite, ite_pred)
    return ate_pred, mae, mae_std

def naive_ipw_nrmse(y: np.ndarray, t: np.ndarray, true_ite: np.array) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ps = np.mean(t)
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    return ate_pred, nrmse(true_ite, ite_pred)

def naive_ipw_mae(y: np.ndarray, t: np.ndarray, true_ite: np.array) -> tuple:
    """MAE with naive inverse propensity weighting"""
    ps = np.mean(t)
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    mae, mae_std = mae_with_std(true_ite, ite_pred)
    return ate_pred, mae, mae_std


def aipw_nrmse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray, true_ite: np.array) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    ite_pred = (t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0))
    ate_pred = np.mean(ite_pred)
    return ate_pred, nrmse(true_ite, ite_pred)

def aipw_mae(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray, true_ite: np.array) -> tuple:
    """MAE with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    ite_pred = (t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0))
    ate_pred = np.mean(ite_pred)
    mae, mae_std = mae_with_std(true_ite, ite_pred)
    return ate_pred, mae, mae_std

def ipw_nrmse_stabilized(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ite: np.ndarray) -> float:
    """Mean-squared-error with inverse propensity weighting and stabilized weights"""
    # Proportion of treatment and control groups
    prop_treatment = np.mean(t)
    prop_control = np.mean(1 - t)

    # Compute stabilized weights
    weights_treatment = prop_treatment / ps
    weights_control = prop_control / (1 - ps)

    # Calculate ITE predictions
    ite_pred = (t * y * weights_treatment) - ((1 - t) * y * weights_control)
    ate_pred = np.mean(ite_pred)

    return ate_pred, nrmse(true_ite, ite_pred)

def ipw_mae_stabilized(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ite: np.ndarray) -> tuple:
    """MAE with inverse propensity weighting and stabilized weights"""
    # Proportion of treatment and control groups
    prop_treatment = np.mean(t)
    prop_control = np.mean(1 - t)

    # Compute stabilized weights
    weights_treatment = prop_treatment / ps
    weights_control = prop_control / (1 - ps)

    # Calculate ITE predictions
    ite_pred = (t * y * weights_treatment) - ((1 - t) * y * weights_control)
    ate_pred = np.mean(ite_pred)
    mae, mae_std = mae_with_std(true_ite, ite_pred)
    return ate_pred, mae, mae_std


def calculate_test_metrics_acic(
    predictions: np.array,
    ite: np.array,
    prefix: str,
    estimator: str,
    eval_metric: str = "nrmse"
) -> Dict[str, float]:
    if estimator == "g-formula":
        ate_std, std_nrmse_ = std_nrmse(predictions['pred_y_A0'], predictions['pred_y_A1'], ite)
        ate_std_mae, std_mae_, std_mae_std = std_mae(predictions['pred_y_A0'], predictions['pred_y_A1'], ite)
        result = {f"{prefix}: predicted ATE for standardization": ate_std,
                  f"{prefix}: NRMSE for standardization": std_nrmse_,
                  f"{prefix}: MAE for standardization": std_mae_,
                  f"{prefix}: MAE std for standardization": std_mae_std}
    elif estimator == "naive ipw":
        ate_naive_ipw, naive_ipw_nrmse_ = naive_ipw_nrmse(predictions['y'], predictions['t'], ite)
        ate_naive_ipw_mae, naive_ipw_mae_, naive_ipw_mae_std = naive_ipw_mae(predictions['y'], predictions['t'], ite)
        result = {f"{prefix}: predicted ATE for naive IPW": ate_naive_ipw,
                  f"{prefix}: NRMSE for naive IPW": naive_ipw_nrmse_,
                  f"{prefix}: MAE for naive IPW": naive_ipw_mae_,
                  f"{prefix}: MAE std for naive IPW": naive_ipw_mae_std}
    elif estimator == "ipw":
        ate_ipw, ipw_nrmse_ = ipw_nrmse(predictions['y'], predictions['t'], predictions['t_prob'], ite)
        ate_ipw_mae, ipw_mae_, ipw_mae_std = ipw_mae(predictions['y'], predictions['t'], predictions['t_prob'], ite)
        result = {f"{prefix}: predicted ATE for IPW": ate_ipw,
                  f"{prefix}: NRMSE for IPW": ipw_nrmse_,
                  f"{prefix}: MAE for IPW": ipw_mae_,
                  f"{prefix}: MAE std for IPW": ipw_mae_std}
    elif estimator == "ipw_stable":
        ate_ipw_stable, ipw_nrmse_stable_ = ipw_nrmse_stabilized(predictions['y'], predictions['t'], predictions['t_prob'], ite)
        ate_ipw_stable_mae, ipw_stable_mae_, ipw_stable_mae_std = ipw_mae_stabilized(predictions['y'], predictions['t'], predictions['t_prob'], ite)
        result = {f"{prefix}: predicted ATE for IPW with stabilized weights": ate_ipw_stable,
                  f"{prefix}: NRMSE for IPW with stabilized weights": ipw_nrmse_stable_,
                  f"{prefix}: MAE for IPW with stabilized weights": ipw_stable_mae_,
                  f"{prefix}: MAE std for IPW with stabilized weights": ipw_stable_mae_std}
    else:
        ate_aipw, aipw_nrmse_ = aipw_nrmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                                           predictions['pred_y_A1'], predictions['t_prob'], ite)
        ate_aipw_mae, aipw_mae_, aipw_mae_std = aipw_mae(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                                           predictions['pred_y_A1'], predictions['t_prob'], ite)
        result = {f"{prefix}: predicted ATE for AIPW": ate_aipw,
                  f"{prefix}: NRMSE for AIPW": aipw_nrmse_,
                  f"{prefix}: MAE for AIPW": aipw_mae_,
                  f"{prefix}: MAE std for AIPW": aipw_mae_std,
                  f"{prefix}: mean of t_prob among treated": np.mean(predictions['t_prob'][predictions['t'] == 1]),
                  f"{prefix}: mean of t_prob among control": np.mean(predictions['t_prob'][predictions['t'] == 0])}
    return result

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument("--estimator", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    args = parser.parse_args()

    pred_g_formula = pd.read_csv(f"experiments/predict/{args.data_name}/predictions_g_formula_sample{args.sample_id}.csv")
    pred_ipw = pd.read_csv(f"experiments/predict/{args.data_name}/predictions_ipw_sample{args.sample_id}.csv")

    ate_aipw, aipw_nrmse = aipw_nrmse(pred_g_formula['y'], pred_g_formula['t'], pred_g_formula['pred_y_A0'],
                          pred_g_formula['pred_y_A1'], pred_ipw['t_prob'])

    # print results
    print(f"Predicted ATE for AIPW (Sep): {ate_aipw}")
    print(f"NRMSE for AIPW (Sep): {aipw_nrmse}")