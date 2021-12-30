import torch

from upfirdn2d.naiive.upfirdn2d import upfirdn2d_naiive
from upfirdn2d.original.upfirdn2d import upfirdn2d_original

random_img = torch.rand((10, 1, 1024, 1024), dtype=torch.float32, device="cuda")

k = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device="cuda")
k = k[None, :] * k[:, None]

i1 = upfirdn2d_naiive(random_img, k, up=1, down=2, pad=(0, 0))
i2 = upfirdn2d_original(random_img, k, up=1, down=2, pad=(0, 0))


print(f" CHECK {'SUCCEEDED' if i1.allclose(i2) else 'FAILED'} ".center(80, "-"))
