"""PyTorch behind a small interface, for the tests of gradients, and tensors
that carry gradients. A test takes it as the ``fw`` fixture, which skips the
test where PyTorch is not installed.

``tensor`` makes a float64 framework array, and ``from_numpy`` one of the
dtype of a NumPy array.
"""

import numpy as np

from stablebear._tracked import tracked_class


class _Torch:
    def __init__(self):
        import torch

        self.torch = torch
        self.tensor_type = torch.Tensor

    def tensor(self, x):
        return self.torch.tensor(x, dtype=self.torch.float64)

    def from_numpy(self, x):
        return self.torch.from_numpy(x)

    def to_numpy(self, x):
        return x.detach().numpy()

    def sum(self, x):
        return x.sum()

    def concat(self, xs):
        return self.torch.cat(xs)

    def value_and_grad(self, f, x0):
        x = self.tensor(x0).requires_grad_()
        y = f(x)
        y.backward()
        assert x.grad is not None
        return y.item(), x.grad.numpy()



FRAMEWORKS = {"torch": _Torch}


def tracked(fw, tensor):
    """The tensor as one that carries gradients of the framework: each group
    of columns of its serial content (the bars of barcodes; the times and
    the values of PCFs; the points of point clouds; the entries of
    matrices) becomes a framework array."""
    content, enumeration, *sizes = tensor._serial()
    parts, start = [], 0
    for _, width in type(tensor)._tracking.groups:
        width = content.shape[1] if width is None else width
        columns = content[:, start] if width == 1 else content[:, start:start + width]
        parts.append(fw.from_numpy(np.ascontiguousarray(columns)))
        start += width
    return tracked_class(type(tensor))(parts, enumeration, sizes=sizes[0] if sizes else None)


def assert_gradient_matches_finite_differences(fw, f, x0, eps=1e-6):
    x0 = np.asarray(x0, dtype=np.float64)
    _, grad = fw.value_and_grad(f, x0)
    numerical = np.zeros_like(x0)
    for i in np.ndindex(x0.shape):
        step = np.zeros_like(x0)
        step[i] = eps
        numerical[i] = (fw.to_numpy(f(fw.tensor(x0 + step))) - fw.to_numpy(f(fw.tensor(x0 - step)))) / (2 * eps)
    np.testing.assert_allclose(grad, numerical, rtol=1e-6, atol=1e-8)
