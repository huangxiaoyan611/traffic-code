import numpy as np
import torch


def _extended_metrics_numpy(pred, true, threshold=5.0):
    """Return the legacy masked metrics plus transparent sample counts.

    The original project reports MAE/MAPE only on targets strictly greater
    than 5.  That protocol is kept for comparability.  RMSE is reported on
    the same valid population and additionally on all samples so that large
    errors on low-flow/zero-flow samples are not hidden.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")

    valid = true > threshold
    diff = pred - true
    valid_diff = diff[valid]
    valid_true = true[valid]

    if valid_diff.size:
        mae = float(np.mean(np.abs(valid_diff)))
        mape = float(np.mean(np.abs(valid_diff / valid_true)))
        rmse = float(np.sqrt(np.mean(valid_diff ** 2)))
    else:
        mae = mape = rmse = float("nan")

    return {
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
        "rmse_all": float(np.sqrt(np.mean(diff ** 2))),
        "zero_count": int(np.count_nonzero(true == 0)),
        "le5_count": int(np.count_nonzero(true <= threshold)),
        "valid_count": int(np.count_nonzero(valid)),
        "total_count": int(true.size),
        "threshold": float(threshold),
    }


def test_metrics_extended(pred, true, threshold=5.0):
    """Evaluate MAE/MAPE/RMSE with the project's explicit >5 mask.

    MAPE never divides by zero because it is evaluated only where the ground
    truth is strictly larger than ``threshold``.  The returned counts make
    the excluded zero and low-flow samples auditable.
    """
    if isinstance(pred, torch.Tensor):
        if not isinstance(true, torch.Tensor):
            raise TypeError("pred and true must have the same type")
        return _extended_metrics_numpy(
            pred.detach().cpu().numpy(), true.detach().cpu().numpy(), threshold
        )
    if isinstance(pred, np.ndarray):
        if not isinstance(true, np.ndarray):
            raise TypeError("pred and true must have the same type")
        return _extended_metrics_numpy(pred, true, threshold)
    raise TypeError(f"Unsupported metric input type: {type(pred)}")

def mae_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def mape_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def mae_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(pred-true))

def mape_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def test_metrics(pred, true, mask1=5, mask2=5):
    # mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = mae_np(pred, true, mask1)
        mape = mape_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae  = mae_torch(pred, true, mask1).item()
        mape = mape_torch(pred, true, mask2).item()
    else:
        raise TypeError
    return mae, mape

