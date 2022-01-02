import torch
from utils.profiler import profile

from upfirdn2d.new.upfirdn2d import upfirdn2d as new
from upfirdn2d.original.upfirdn2d import upfirdn2d as original

# Create random FP image
random_img = torch.rand((8, 1, 1024, 1024), dtype=torch.float32, device="cuda")

# Create kernel
k = torch.tensor([1, 3, 3, 1], dtype=torch.float32, device="cuda")
k = k[None, :] * k[:, None]
k /= k.sum()


# Compare output from original and modified kernels
def fn(i: int) -> None:
    original_result = original(random_img, k, up=2, down=1, pad=(0, 0))
    new_result = new(random_img, k, up=2, down=1, pad=(0, 0), force_generic=False)
    new_result_generic = new(
        random_img, k, up=2, down=1, pad=(0, 0), force_generic=True
    )
    assert new_result.allclose(original_result), f"Assertion failed at iteration {i}"
    assert new_result_generic.allclose(new_result), f"Assertion failed at iteration {i}"


profile(fn, ["upfirdn2d"], iters=5000)
