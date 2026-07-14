"""Duck-typed array conversion so report/viz never import a framework."""
import numpy as np


def to_numpy(x):
    """torch.Tensor (any device), np.ndarray, or array-like -> np.ndarray."""
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    if detach is not None:                 # torch tensor
        x = detach()
        cpu = getattr(x, "cpu", None)
        if cpu is not None:
            x = cpu()
        return x.numpy()
    return np.asarray(x)
