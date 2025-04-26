import torch
from sklearn.metrics import mean_squared_error, r2_score


def multitarget_r2_avg(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    y_hat_np = y_hat.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # reshape (batch, 1) to (batch,) for single-target
    if y_np.ndim == 2 and y_np.shape[1] == 1:
        y_np = y_np.squeeze(1)
        y_hat_np = y_hat_np.squeeze(1)

    return r2_score(y_np, y_hat_np, multioutput="uniform_average")


def multitarget_mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    y_hat_np = y_hat.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    if y_np.ndim == 2 and y_np.shape[1] == 1:
        y_np = y_np.squeeze(1)
        y_hat_np = y_hat_np.squeeze(1)

    return mean_squared_error(y_np, y_hat_np)
