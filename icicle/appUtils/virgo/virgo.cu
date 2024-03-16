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
    __global__ void mul_pair_kernel(S* arr1, S* arr2, S* result, S inv_r_mont2, int n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    result[tid] = arr1[tid] * arr2[tid] * inv_r_mont2;
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
  __global__ void reduce_sum_kernel2(S** result, uint32_t arr_count, uint32_t n, uint32_t half) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t arr_id = tid % arr_count;
    uint32_t arr_pos = tid / arr_count;

    auto other = arr_pos + half;
    if (other < n) {
      result[arr_id][arr_pos] = result[arr_id][arr_pos] + result[arr_id][other];
    }
  }

  template <typename S>
  cudaError_t sum_arrays(S** arrs, int arr_count, int n)
  {
    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto x = n;
    while (x > 1) {
      int worker_count = (x * arr_count) >> 1;
      int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
      int num_blocks = (worker_count + num_threads - 1) / num_threads;

      int half = (x + 1) >> 1;
      reduce_sum_kernel2 <<< num_blocks, num_threads, 0, stream >>> (arrs, arr_count, x, half);

      x = (x + 1) >> 1;
    }
  }

  template <typename S>
  cudaError_t sum_single_array(S* arr, int n)
  {
    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto x = n;
    while (x > 1) {
      int worker_count = x >> 1;
      int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
      int num_blocks = (worker_count + num_threads - 1) / num_threads;

      int half = (x + 1) >> 1;
      reduce_sum_kernel <<< num_blocks, num_threads, 0, stream >>> (arr, x, half);

      x = (x + 1) >> 1;
    }
  }

  template <typename S>
  cudaError_t bk_sum_all_case1(
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

    // This is arkwork inverse R, not icicle inverse R.
    // inv_r = 9915499612839321149637521777990102151350674507940716049588462388200839649614
    // S inv_r_mont({0x6db1194e, 0xdc5ba005, 0xe111ec87, 0x90ef5a9, 0xaeb85d5d, 0xc8260de4, 0x82c5551c, 0x15ebf951});
    // inv_r2 = inv_r ^ 2 = 8519677608991584271437967308266649112183478179623991153221810821821888926024
    S inv_r_mont2({0xd3c71148, 0xae12ba81, 0xb38e2428, 0x52f28270, 0x79a1edeb, 0xe065f3e3, 0xe436631e, 0x12d5f775});

    mul_pair_kernel <<< num_blocks, num_threads, 0, stream >>> (arr1, arr2, device_tmp, inv_r_mont2, n);

    // auto x = n;
    // while (x > 1) {
    //   int worker_count = x >> 1;
    //   int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
    //   int num_blocks = (worker_count + num_threads - 1) / num_threads;

    //   int half = (x + 1) >> 1;
    //   reduce_sum_kernel <<< num_blocks, num_threads, 0, stream >>> (device_tmp, x, half);

    //   x = (x + 1) >> 1;
    // }

    sum_single_array(device_tmp, n);

    cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyDeviceToHost);

    cudaFree(device_tmp);

    return CHK_LAST();
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkSumAllCase1)(
    curve_config::scalar_t* arr1,
    curve_config::scalar_t* arr2,
    curve_config::scalar_t* output,
    int n)
  {
    return bk_sum_all_case1<curve_config::scalar_t>(arr1, arr2, output, n);
  }
}
