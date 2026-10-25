#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* values,
    const unsigned int* prefix_sum_accum,
          unsigned int* output,
    unsigned int n,
    unsigned int bit_offset)
{
    __shared__ unsigned int digit_offset[RADIX];
    __shared__ unsigned int chunk_digits[GROUP_SIZE];
    __shared__ unsigned int chunk_counts[RADIX];

    const unsigned int tid = threadIdx.x;
    const unsigned int group_id = blockIdx.x;
    const unsigned int num_groups = gridDim.x;

    if (tid < RADIX) {
        const unsigned int idx = tid * num_groups + group_id;
        digit_offset[tid] = (idx == 0) ? 0 : prefix_sum_accum[idx - 1];
    }

    const unsigned int tile_begin = group_id * TILE_SIZE;
    for (unsigned int k = 0; k < ELEMS_PER_THREAD; ++k) {
        if (tid < RADIX)
            chunk_counts[tid] = 0;

        const unsigned int i = tile_begin + k * GROUP_SIZE + tid;
        const bool valid = (i < n);
        unsigned int value = 0;
        unsigned int digit = RADIX;
        if (valid) {
            value = values[i];
            digit = (value >> bit_offset) & (RADIX - 1);
        }
        chunk_digits[tid] = digit;
        __syncthreads();

        if (valid) {
            unsigned int before_in_chunk = 0;
            for (unsigned int j = 0; j < tid; ++j) {
                if (chunk_digits[j] == digit)
                    ++before_in_chunk;
            }
            output[digit_offset[digit] + before_in_chunk] = value;
            atomicAdd(&chunk_counts[digit], 1);
        }
        __syncthreads();

        if (tid < RADIX)
            digit_offset[tid] += chunk_counts[tid];
        __syncthreads();
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
