#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

static __host__ __device__ __forceinline__ int floor_div(int a, int b) {
	int c = a / b;

	if (c * b > a) {
		c--;
	}

	return c;
}

struct UpFirDn2DKernelParams {
	int up;
	int down;
	int pad_0;
	int pad_1;
	int n;
	int in_h;
	int in_w;
	int kernel_h;
	int kernel_w;
	int out_h;
	int out_w;
};

/**
 * @brief Applies upfirdn operation to input of shape (n, in_h, in_w)
 *
 * Parameters:
 * - output size: (n, out_h, out_w) where out_h > in_h and out_w > in_w
 * - kernel size: (kernel_h, kernel_w)
 * - block size (i.e. no. of threads): (block_out_h, block_out_w)
 * - grid size (i.e. no. of blocks): (ceil(out_h / block_out_h), ceil(out_w / block_out_w), n)
 *
 * @tparam scalar_t
 * @tparam up
 * @tparam down
 * @tparam kernel_h
 * @tparam kernel_w
 * @tparam block_out_h
 * @tparam block_out_w
 * @param out
 * @param input
 * @param kernel
 * @param p
 * @return __global__
 */
template <typename scalar_t, int up, int down, int kernel_h, int kernel_w, int block_out_h, int block_out_w>
__global__ void upfirdn2d_kernel(scalar_t *out, const scalar_t *input, const scalar_t *kernel, const UpFirDn2DKernelParams p) {
	// blockDim.x == block_out_w, blockDim.y == block_out_h

	// Size of input tile to obtain desired size of output tile (reverse of the previous step in calculating overall output size)
	constexpr int block_in_h = ((block_out_h - 1) * down + kernel_h - 1) / up + 1;
	constexpr int block_in_w = ((block_out_w - 1) * down + kernel_w - 1) / up + 1;

	// Shared kernel and shared x (input)
	__shared__ volatile float sk[kernel_h][kernel_w];
	__shared__ volatile float sx[block_in_h][block_in_w];

	// Block Information (no checks on block_out_x < p.out_w cuz it will be checked on host)
	int block_out_y = blockIdx.x * block_out_h;	 // Starting y-coordinate of block output
	int block_out_x = blockIdx.y * block_out_w;	 // Starting x-coordinate of block output
	int block_n		= blockIdx.z;

	// Thread Information (x and y does not really matter)
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	// Load shared kernel
	if (thread_x < p.kernel_w & thread_y < p.kernel_h) {
		sk[ky][kx] = kernel[(p.kernel_h - 1 - thread_y) * p.kernel_w + (p.kernel_w - 1 - thread_x)];
	}

	// int block_mid_x = block_out_x_base * down + up - 1 - p.pad_0;
	// int block_mid_y = block_out_y_base * down + up - 1 - p.pad_0;
	// int block_in_x	= floor_div(block_mid_x, up);
	// int block_in_y	= floor_div(block_mid_y, up);

	__syncthreads();

	// Load shared input
	for (int i = 0;) {
		int in_x = thread_x + block_in_x;
		int in_y = thread_y + block_in_y;

		scalar_t v = 0.0;

		if (in_x >= 0 & in_y >= 0 & in_x < p.in_w & in_y < p.in_h) {
			sx[thread_y][thread_x] = input[(block_n * p.in_h + in_y) * p.in_w + in_x];
		}
	}

	// Calculate output
	__syncthreads();
	int out_x = thread_x + block_out_x;
	int out_y = thread_y + block_out_y;

	int mid_x	 = block_mid_x + thread_x * down;
	int mid_y	 = block_mid_y + thread_y * down;
	int in_x	 = floor_div(mid_x, up);
	int in_y	 = floor_div(mid_y, up);
	int rel_in_x = in_x - block_in_x;
	int rel_in_y = in_y - block_in_y;
	int kernel_x = (in_x + 1) * up_x - mid_x - 1;
	int kernel_y = (in_y + 1) * up - mid_y - 1;

	scalar_t v = 0.0;

#pragma unroll
	for (int y = 0; y < kernel_h / up; y++)
#pragma unroll
		for (int x = 0; x < kernel_w / up_x; x++)
			v += sx[rel_in_y + y][rel_in_x + x] *
				 sk[kernel_y + y * up][kernel_x + x * up_x];

	if (out_x < p.out_w & out_y < p.out_h) {
		out[((major_idx * p.out_h + out_y) * p.out_w + out_x) * p.minor_dim +
			minor_idx] = v;
	}
}

// Entrypoint of CUDA operation
torch::Tensor upfirdn2d_op(const torch::Tensor &input, const torch::Tensor &kernel, int up, int down, int pad_0, int pad_1) {
	// Get current thread's CUDA device (set by DeviceGuard) and stream
	int curDevice = -1;
	cudaGetDevice(&curDevice);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	// Seems extra given that input is already verified to be contiguous
	auto x = input.contiguous();
	auto k = kernel.contiguous();

	// N * C, H, W, 1
	UpFirDn2DKernelParams p;
	p.n		   = x.size(0);
	p.in_h	   = x.size(1);
	p.in_w	   = x.size(2);
	p.kernel_h = k.size(0);
	p.kernel_w = k.size(1);
	p.up	   = up;
	p.down	   = down;
	p.pad_0	   = pad_0;
	p.pad_1	   = pad_1;

	// Out = Ceil((In * Upsample + Paddings - (Kernel - 1)) / Downsample)
	p.out_h = (p.in_h * p.up + p.pad_0 + p.pad_1 - p.kernel_h + p.down) / p.down;
	p.out_w = (p.in_w * p.up + p.pad_0 + p.pad_1 - p.kernel_w + p.down) / p.down;

	// Options: dtype, device, layout
	auto out = at::empty({p.n, p.out_h, p.out_w}, x.options());

	// Selecting upfirdn mode and tile size based on options
	int block_out_h = 32;
	int block_out_w = 32;

	dim3 block_size;
	dim3 grid_size;

	if (block_out_h > 0 && block_out_w > 0) {
		/**
		 * Arrangement:
		 *    Each thread responsible for tile of output size (block_out_h, block_out_w)
		 *
		 *    No. of blocks:
		 *    (
		 * 	    x: loop over height and minor_dim
		 *      y: loop over width
		 *      z: loop over major_dim, max is 16384
		 *         (if more, p.loop_major accounts for loop)
		 *    )
		 *
		 *    No. of threads: (256, 1, 1)
		 *
		 */

		// ceil(N * C / 16384)
		p.loop_major = (p.major_dim - 1) / 16384 + 1;
		p.loop_x	 = 1;
		// 256 threads per block
		block_size = dim3(32 * 8, 1, 1);
		// ceil(out_h / block_out_h)
		// ceil(out_w / block_out_w)
		// ceil(N * C / ceil(N * C / 16384))
		grid_size = dim3(((p.out_h - 1) / block_out_h + 1) * p.minor_dim,
						 (p.out_w - 1) / (p.loop_x * block_out_w) + 1,
						 (p.major_dim - 1) / p.loop_major + 1);
	} else {
		// Why need loop?
		p.loop_major = (p.major_dim - 1) / 16384 + 1;
		p.loop_x	 = 4;
		block_size	 = dim3(4, 32, 1);
		grid_size	 = dim3((p.out_h * p.minor_dim - 1) / block_size.x + 1,
							(p.out_w - 1) / (p.loop_x * block_size.y) + 1,
							(p.major_dim - 1) / p.loop_major + 1);
	}

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&] {
		switch (mode) {
		case 1:
			upfirdn2d_kernel<scalar_t, 1, 1, 1, 1, 4, 4, 16, 64>
				<<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
													   x.data_ptr<scalar_t>(),
													   k.data_ptr<scalar_t>(), p);

			break;

		case 2:
			upfirdn2d_kernel<scalar_t, 1, 1, 1, 1, 3, 3, 16, 64>
				<<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
													   x.data_ptr<scalar_t>(),
													   k.data_ptr<scalar_t>(), p);

			break;

		case 3:
			upfirdn2d_kernel<scalar_t, 2, 2, 1, 1, 4, 4, 16, 64>
				<<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
													   x.data_ptr<scalar_t>(),
													   k.data_ptr<scalar_t>(), p);

			break;

		case 4:
			upfirdn2d_kernel<scalar_t, 2, 2, 1, 1, 2, 2, 16, 64>
				<<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
													   x.data_ptr<scalar_t>(),
													   k.data_ptr<scalar_t>(), p);

			break;

		case 5:
			upfirdn2d_kernel<scalar_t, 1, 1, 2, 2, 4, 4, 8, 32>
				<<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
													   x.data_ptr<scalar_t>(),
													   k.data_ptr<scalar_t>(), p);

			break;

		case 6:
			upfirdn2d_kernel<scalar_t, 1, 1, 2, 2, 4, 4, 8, 32>
				<<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
													   x.data_ptr<scalar_t>(),
													   k.data_ptr<scalar_t>(), p);

			break;

		default:
			upfirdn2d_kernel_large<scalar_t><<<grid_size, block_size, 0, stream>>>(
				out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
				k.data_ptr<scalar_t>(), p);
		}
	});

	return out;
}