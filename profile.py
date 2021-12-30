import torch

from upfirdn2d.naiive.upfirdn2d import upfirdn2d_naiive
from upfirdn2d.original.upfirdn2d import upfirdn2d_original
from utils.profiler import profile

random_img = torch.rand((10, 1, 1024, 1024), dtype=torch.float32, device="cuda")

k = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device="cuda")
k = k[None, :] * k[:, None]


def fn(i: int):
    i1 = upfirdn2d_naiive(random_img, k, up=1, down=2, pad=(0, 0))
    i2 = upfirdn2d_original(random_img, k, up=1, down=2, pad=(0, 0))
    assert i1.allclose(i2), f"Assertion failed at iteration {i}"


profile(fn, ["upfirdn2d"], iters=1000)
