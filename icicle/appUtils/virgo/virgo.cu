#include "virgo.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <string>
#include <sstream>

namespace virgo {
  template <typename S>
    __global__ void mul_pair_kernel(S* arr1, S* arr2, S* result, int n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    result[tid] = arr1[tid] * arr2[tid];
  }

  template <typename S>
  __global__ void reduce_sum_kernel(S* result, uint32_t n, uint32_t half) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto other = tid + half;
    if (other < n) {
      result[tid] = result[tid] + result[other];
    }
  }

  template <typename S>
  cudaError_t sumcheck_sum(
    S* arr1, S* arr2, S* output, int n)
  {
    CHK_INIT_IF_RETURN();

    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);

    int worker_count = n;
    int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
    int num_blocks = (worker_count + num_threads - 1) / num_threads;

    const int log_n = log2(n);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    S* device_tmp;
    // allocate device array
    cudaMalloc((void**)&device_tmp, n * sizeof(S));

    // std::cout << "num_blocks, num_threads = " << num_blocks << " " << num_threads << std::endl;

    mul_pair_kernel <<< num_blocks, num_threads, 0, stream >>> (arr1, arr2, device_tmp, n);

    auto x = n;
    while (x > 1) {
      int worker_count = x >> 1;
      int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
      int num_blocks = (worker_count + num_threads - 1) / num_threads;

      int half = (x + 1) >> 1;
      reduce_sum_kernel <<< num_blocks, num_threads, 0, stream >>> (device_tmp, x, half);

      x = (x + 1) >> 1;
    }

    cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyDeviceToHost);

    cudaFree(device_tmp);

    return CHK_LAST();
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, SumcheckSum)(
    curve_config::scalar_t* arr1,
    curve_config::scalar_t* arr2,
    curve_config::scalar_t* output,
    int n)
  {
    return sumcheck_sum<curve_config::scalar_t>(arr1, arr2, output, n);
  }
}
