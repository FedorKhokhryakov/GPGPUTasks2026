#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* values,
          unsigned int* counts,
    unsigned int n,
    unsigned int bit_offset)
{
    __shared__ unsigned int local_counts[RADIX];

    const unsigned int tid = threadIdx.x;
    const unsigned int group_id = blockIdx.x;
    const unsigned int num_groups = gridDim.x;

    if (tid < RADIX)
        local_counts[tid] = 0;
    __syncthreads();

    const unsigned int tile_begin = group_id * TILE_SIZE;
    for (unsigned int k = 0; k < ELEMS_PER_THREAD; ++k) {
        const unsigned int i = tile_begin + k * GROUP_SIZE + tid;
        if (i < n) {
            const unsigned int digit = (values[i] >> bit_offset) & (RADIX - 1);
            atomicAdd(&local_counts[digit], 1);
        }
    }
    __syncthreads();

    if (tid < RADIX)
        counts[tid * num_groups + group_id] = local_counts[tid];
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
