import torch

from upfirdn2d.new.upfirdn2d import upfirdn2d as new
from upfirdn2d.original.upfirdn2d import upfirdn2d as original

# Create random FP image
random_img = torch.rand((1, 1, 8, 8), dtype=torch.float32, device="cuda")

# Create kernel
k = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device="cuda")
k = k[None, :] * k[:, None]

# Check results
original_result = original(random_img, k, up=2, down=1, pad=(0, 0))
print(original_result)
new_result = new(random_img, k, up=2, down=1, pad=(0, 0))
print(new_result)

print(f" CHECK {'SUCCEEDED' if new_result.allclose(i1) else 'FAILED'} ".center(80, "-"))
