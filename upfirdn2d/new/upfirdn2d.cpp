#include "upfirdn2d.hpp"

#include <ATen/ATen.h>
#include <torch/extension.h>

// Python function definition
torch::Tensor upfirdn2d(
	const torch::Tensor &input, const torch::Tensor &kernel,
	int up_x, int up_y, int down_x, int down_y, int pad_x0,
	int pad_x1, int pad_y0, int pad_y1) {
	// Check input validity
	CHECK_INPUT(input);
	CHECK_INPUT(kernel);

	// Sets default CUDA device, and reset it when destructed.
	at::DeviceGuard guard(input.device());

	return upfirdn2d_op(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1,
						pad_y0, pad_y1);
}

// Defines function upfirdn2d inside Pytorch extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)");
}