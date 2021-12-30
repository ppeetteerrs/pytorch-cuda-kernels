#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

/**
 * Using macros (((a) + (b) - 1) / (b)) or (((a) - 1) / (b) + 1) are wrong for non-positive numbers
 * Note that these macros are also not identical in behaviour, the former is "less wrong"
 **/
static __host__ __device__ __forceinline__ int ceil_div(int a, int b) {
	int c = a / b;

	if (c * b < a) {
		c++;
	}

	return c;
}

struct UpFirDn2DKernelParams {
	int up_x;
	int up_y;
	int down_x;
	int down_y;
	int pad_x0;
	int pad_x1;
	int pad_y0;
	int pad_y1;

	int n;
	int in_h;
	int in_w;
	int kernel_h;
	int kernel_w;
	int out_h;
	int out_w;
	int loop_n;
};

/**
 * Each thread will handle pixels:
 * DIM n: id.z * loop_n to (id.z + 1) * loop_n - 1
 * DIM x: block.x + id.x
 * DIM y: block.y + id.y
 *
 **/
template <typename scalar_t>
__global__ void upfirdn2d_kernel(scalar_t *out, const scalar_t *input,
								 const scalar_t				*kernel,
								 const UpFirDn2DKernelParams p) {
	// Output pixel(s) (base) coordinates
	const int out_x		 = blockIdx.x * blockDim.x + threadIdx.x;
	const int out_y		 = blockIdx.y * blockDim.y + threadIdx.y;
	const int out_n_base = blockIdx.z * p.loop_n;

	if (out_x >= p.out_w || out_y >= p.out_h || out_n_base >= p.n) {
		return;
	}

	// Calculate middle layer (after upsampling) coordinates
	const int mid_x = out_x * p.down_x - p.pad_x0;
	const int mid_y = out_y * p.down_y - p.pad_y0;

	const int in_x	   = min(max(ceil_div(mid_x, p.up_x), 0), p.in_w);
	const int w		   = min(max(ceil_div(mid_x + p.kernel_w, p.up_x), 0), p.in_w) - in_x;
	const int kernel_x = p.kernel_w - 1 + mid_x - in_x * p.up_x;

	const int in_y	   = min(max(ceil_div(mid_y, p.up_y), 0), p.in_h);
	const int h		   = min(max(ceil_div(mid_y + p.kernel_h, p.up_y), 0), p.in_h) - in_y;
	const int kernel_y = p.kernel_h - 1 + mid_y - in_y * p.up_y;

	// Loop over DIM N
	for (int loop_n = 0, out_n = out_n_base; loop_n < p.loop_n && out_n < p.n; loop_n++, out_n++) {
		// Pointer to start of input and kernel
		const scalar_t *x_p = &input[(out_n * p.in_h + in_y) * p.in_w + in_x];
		const scalar_t *k_p = &kernel[kernel_y * p.kernel_w + kernel_x];

		// Pointer step sizes in DIM x
		const int x_px = 1;
		const int k_px = -p.up_x;

		// Pointer step sizes to move from (end_x, y) to (start_x, y+1)
		const int x_py = p.in_w - w * x_px;
		const int k_py = -p.up_y * p.kernel_w - w * k_px;

		scalar_t v = 0.0f;

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				// Accumulate sum-product
				v += static_cast<scalar_t>(*x_p) * static_cast<scalar_t>(*k_p);
				// Move pointer in x-direction
				x_p += x_px;
				k_p += k_px;
			}

			x_p += x_py;
			k_p += k_py;
		}

		// Store output pixel
		out[(out_n * p.out_h + out_y) * p.out_w + out_x] = v;
	}
}

/**
 * input: (N, 1, H, W)
 * kernel: (KH, KW)
 **/
torch::Tensor upfirdn2d_op(const torch::Tensor &input,
						   const torch::Tensor &kernel, int up_x, int up_y,
						   int down_x, int down_y, int pad_x0, int pad_x1,
						   int pad_y0, int pad_y1) {
	// int curDevice = -1;
	// cudaGetDevice(&curDevice);
	// Get CUDA stream
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	UpFirDn2DKernelParams p;

	auto x = input.contiguous();
	auto k = kernel.contiguous();

	p.n		   = x.size(0);
	p.in_h	   = x.size(2);
	p.in_w	   = x.size(3);
	p.kernel_h = k.size(0);
	p.kernel_w = k.size(1);
	p.up_x	   = up_x;
	p.up_y	   = up_y;
	p.down_x   = down_x;
	p.down_y   = down_y;
	p.pad_x0   = pad_x0;
	p.pad_x1   = pad_x1;
	p.pad_y0   = pad_y0;
	p.pad_y1   = pad_y1;

	// out_dim = ceil((in_dim * upsample + paddings - (kernel_dim - 1)) / downsample)
	p.out_h = ceil_div(p.in_h * p.up_y + p.pad_y0 + p.pad_y1 - (p.kernel_h - 1), p.down_y);
	p.out_w = ceil_div(p.in_w * p.up_x + p.pad_x0 + p.pad_x1 - (p.kernel_w - 1), p.down_x);

	// Prepare output tensor
	auto out = at::empty({p.n, 1, p.out_h, p.out_w}, x.options());

	// Number of times to loop in dim_n: ceil(p.n . 16384), i.e. cap at 16384 blocks
	p.loop_n		= max(ceil_div(p.n, 16384), 10);
	auto block_size = dim3(32, 32, 1);
	auto grid_size	= dim3(ceil_div(p.out_w, block_size.x),
						   ceil_div(p.out_h, block_size.y),
						   ceil_div(p.n, p.loop_n));

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&] {
		upfirdn2d_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(
			out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
			k.data_ptr<scalar_t>(), p);
	});

	return out;
}