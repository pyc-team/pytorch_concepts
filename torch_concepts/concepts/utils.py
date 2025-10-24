import torch


def _is_int_index(x) -> bool:
    return isinstance(x, int) or (isinstance(x, torch.Tensor) and x.dim() == 0)


def _check_tensors(tensors):
    B = tensors[0].shape[0]
    dtype = tensors[0].dtype
    device = tensors[0].device
    rest_shape = tensors[0].shape[2:]  # dims >=2 must match
    for i, t in enumerate(tensors):
        if t.dim() < 2:
            raise ValueError(f"Tensor {i} must have at least 2 dims (B, c_i, ...); got {tuple(t.shape)}.")
        if t.shape[0] != B:
            raise ValueError(f"All tensors must share batch dim. Got {t.shape[0]} != {B} at field {i}.")
        # only dim=1 may vary; dims >=2 must match exactly
        if t.shape[2:] != rest_shape:
            raise ValueError(
                f"All tensors must share trailing shape from dim=2. "
                f"Field {i} has {t.shape[2:]} != {rest_shape}."
            )
        if t.dtype != dtype:
            raise ValueError("All tensors must share dtype.")
        if t.device != device:
            raise ValueError("All tensors must be on the same device.")
        if t.requires_grad != tensors[0].requires_grad:
            raise ValueError("All tensors must have the same requires_grad setting.")
