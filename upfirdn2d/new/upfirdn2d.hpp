#include <ATen/ATen.h>
#include <torch/extension.h>

// Settings
#define TILE_OUT_H 32
#define TILE_OUT_W 32
#define N_THREADS  (TILE_OUT_H * TILE_OUT_W)

// Derived
#define N_OUT_LOOP (TILE_OUT_H * TILE_OUT_W / N_THREADS)

// stat
static_assert(N_THREADS <= 1024, "Make sure block size is less than 1024.");
static_assert((TILE_OUT_H * TILE_OUT_W) % N_THREADS == 0, "Make sure tile size is integer multiple of block size.");

// Utility Macros
#define CHECK_CUDA(x)		TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
	CHECK_CUDA(x);     \
	CHECK_CONTIGUOUS(x)

torch::Tensor upfirdn2d_op(
	const torch::Tensor &input, const torch::Tensor &kernel,
	int up_x, int up_y, int down_x, int down_y, int pad_x0,
	int pad_x1, int pad_y0, int pad_y1);

// static_assert(1 == 2);