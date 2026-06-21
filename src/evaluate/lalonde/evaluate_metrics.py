from typing import Dict
import numpy as np
from argparse import ArgumentParser
import pandas as pd
from src.utils import rmse

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2))

def mae_with_std(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Mean Absolute Error with standard deviation."""
    abs_errors = np.abs(y_true - y_pred)
    mae = np.mean(abs_errors)
    mae_std = np.std(abs_errors)
    return mae, mae_std

def std_nrmse(mu0: np.array, mu1: np.array, true_ate=1794.34) -> float:
    """Plug-in estimator, equivalent to standardization."""
    ate_pred = np.mean(mu1 - mu0)
    return ate_pred, nrmse(true_ate, ate_pred)

def std_mae(mu0: np.array, mu1: np.array, true_ate=1794.34) -> tuple:
    """Plug-in estimator with MAE metric.
    For lalonde, we only have true ATE (scalar), so MAE = |predicted_ATE - true_ATE| and std = 0."""
    ate_pred = np.mean(mu1 - mu0)
    mae = np.abs(ate_pred - true_ate)
    mae_std = 0.0  # Only one comparison, so std is 0
    return ate_pred, mae, mae_std


def ipw_nrmse(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ate=1794.34) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    weight_t1 = 1 / (ps+0.01)
    weight_t0 = 1 / (1-(ps+0.01))
    ate_pred = np.mean((t * y * weight_t1) - ((1 - t) * y * weight_t0))
    return ate_pred, nrmse(true_ate, ate_pred)

def ipw_mae(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ate=1794.34) -> tuple:
    """MAE with inverse propensity weighting.
    For lalonde, we only have true ATE (scalar), so MAE = |predicted_ATE - true_ATE| and std = 0."""
    weight_t1 = 1 / (ps+0.01)
    weight_t0 = 1 / (1-(ps+0.01))
    ate_pred = np.mean((t * y * weight_t1) - ((1 - t) * y * weight_t0))
    mae = np.abs(ate_pred - true_ate)
    mae_std = 0.0  # Only one comparison, so std is 0
    return ate_pred, mae, mae_std


def aipw_nrmse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray, true_ate=1794.34) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    ate_pred = np.mean((t * (y - mu1) / (ps+0.01)) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0))
    return ate_pred, nrmse(true_ate, ate_pred)

def aipw_mae(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray, true_ate=1794.34) -> tuple:
    """MAE with Counterfactual Cross Validation, equivalent to doubly robust estimator.
    For lalonde, we only have true ATE (scalar), so MAE = |predicted_ATE - true_ATE| and std = 0."""
    ate_pred = np.mean((t * (y - mu1) / (ps+0.01)) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0))
    mae = np.abs(ate_pred - true_ate)
    mae_std = 0.0  # Only one comparison, so std is 0
    return ate_pred, mae, mae_std

def ipw_nrmse_stabilized(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ate=1794.34) -> float:
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

    return ate_pred, nrmse(true_ate, ate_pred)

def aipw_nrmse_stabilized(y: np.ndarray, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, ps: np.ndarray, true_ate=1794.34) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator, with stabilized weights."""
    # Proportion of treatment and control groups
    prop_treatment = np.mean(t)
    prop_control = np.mean(1 - t)

    # Compute stabilized weights
    weights_treatment = prop_treatment / ps
    weights_control = prop_control / (1 - ps)

    # Calculate ITE predictions
    ite_pred = (t * (y - mu1) * weights_treatment) - ((1 - t) * (y - mu0) * weights_control) + (mu1 - mu0)
    ate_pred = np.mean(ite_pred)

    return ate_pred, nrmse(true_ate, ate_pred)


def calculate_test_metrics(
    predictions: np.array,
    prefix: str,
    estimator: str="ipw",
    eval_metric: str="nrmse"
) -> Dict[str, float]:
    if estimator == "g-formula":
        ate_std, std_nrmse_ = std_nrmse(predictions['pred_y_A0'], predictions['pred_y_A1'])
        ate_std_mae, std_mae_, std_mae_std = std_mae(predictions['pred_y_A0'], predictions['pred_y_A1'])
        result = {f"{prefix}: predicted ATE for standardization": ate_std,
                  f"{prefix}: NRMSE for standardization": std_nrmse_,
                  f"{prefix}: MAE for standardization": std_mae_,
                  f"{prefix}: MAE std for standardization": std_mae_std}
        # Verify ate_std and ate_std_mae are the same (they should be)
    elif estimator == "ipw":
        ate_ipw, ipw_nrmse_ = ipw_nrmse(predictions['y'], predictions['t'], predictions['t_prob'])
        ate_ipw_mae, ipw_mae_, ipw_mae_std = ipw_mae(predictions['y'], predictions['t'], predictions['t_prob'])
        result = {f"{prefix}: predicted ATE for IPW": ate_ipw,
                  f"{prefix}: NRMSE for IPW": ipw_nrmse_,
                  f"{prefix}: MAE for IPW": ipw_mae_,
                  f"{prefix}: MAE std for IPW": ipw_mae_std}
    else:
        ate_aipw, aipw_nrmse_ = aipw_nrmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                          predictions['pred_y_A1'], predictions['t_prob'])
        ate_aipw_mae, aipw_mae_, aipw_mae_std = aipw_mae(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                          predictions['pred_y_A1'], predictions['t_prob'])
        result = {f"{prefix}: predicted ATE for AIPW": ate_aipw,
                  f"{prefix}: NRMSE for AIPW": aipw_nrmse_,
                  f"{prefix}: MAE for AIPW": aipw_mae_,
                  f"{prefix}: MAE std for AIPW": aipw_mae_std}
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument("--estimator", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    args = parser.parse_args()

    pred_g_formula = pd.read_csv(f"experiments/results/{args.data_name}/predictions_g_formula_sample{args.sample_id}.csv")
    pred_ipw = pd.read_csv(f"experiments/results/{args.data_name}/predictions_ipw_sample{args.sample_id}.csv")

    ate_aipw, aipw_nrmse = aipw_nrmse(pred_g_formula['y'], pred_g_formula['t'], pred_g_formula['pred_y_A0'],
                          pred_g_formula['pred_y_A1'], pred_ipw['t_prob'])

    # print results
    print(f"Predicted ATE for AIPW (Sep): {ate_aipw}")
    print(f"NRMSE for AIPW (Sep): {aipw_nrmse}")
