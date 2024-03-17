#include "virgo.cuh"
#include <iostream>
#include <string>
#include <sstream>

namespace virgo {
  // This is arkwork inverse R, not icicle inverse R.
  // inv_r = 9915499612839321149637521777990102151350674507940716049588462388200839649614
  // inv_r2 = inv_r ^ 2 = 8519677608991584271437967308266649112183478179623991153221810821821888926024
  template <typename S>
  __device__ constexpr S inv_r_mont = S({0x6db1194e, 0xdc5ba005, 0xe111ec87, 0x90ef5a9, 0xaeb85d5d, 0xc8260de4, 0x82c5551c, 0x15ebf951});

  template <typename S>
  __device__ constexpr S inv_r_mont2 = S({0xd3c71148, 0xae12ba81, 0xb38e2428, 0x52f28270, 0x79a1edeb, 0xe065f3e3, 0xe436631e, 0x12d5f775});

  /////////////////////////////////
  /// COMMON FUNCTIONS
  /////////////////////////////////

  template <typename S>
  __global__ void mul_pair_kernel(S* arr1, S* arr2, S* result, int n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    result[tid] = arr1[tid] * arr2[tid] * (inv_r_mont<S>);
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
  __global__ void reduce_sum_kernel2(S* result, uint32_t m, uint32_t n, uint32_t half, uint32_t offset) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t arr_id = tid % m;
    uint32_t arr_pos = tid / m;

    auto other = arr_pos + half;
    if (other < n) {
      auto start = offset * arr_id + arr_pos;
      result[start] = result[start] + result[start + half];
    }
  }

  template <typename S>
  cudaError_t sum_arrays(S* arrs, uint32_t m, uint32_t n)
  {
    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto x = n;
    while (x > 1) {
      int worker_count = (x * m + 1) >> 1;
      int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
      int num_blocks = (worker_count + num_threads - 1) / num_threads;

      int half = (x + 1) >> 1;
      reduce_sum_kernel2 <<< num_blocks, num_threads, 0, stream >>> (arrs, m, x, half, n);

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

  /////////////////////////////////
  /// BookKeeping sum_all
  /////////////////////////////////

  template <typename S>
  void print_arr(S* arr, int start, int end) {
    int len = end - start;
    S* tmp = (S*)malloc(len * sizeof(S));

    cudaMemcpy(tmp, arr + start, len * sizeof(S), cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; i++) {
      std::cout << tmp[i] * inv_r_mont<S> << " ";
    }
    std::cout << std::endl;

    delete [] tmp;
  }

  template <typename S>
  cudaError_t bk_sum_all_case_1(
    S* table1, S* table2, S* output, int n)
  {
    CHK_INIT_IF_RETURN();

    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);

    S* device_tmp;
    // allocate device array
    cudaMalloc((void**)&device_tmp, n * sizeof(S));

    // 1. Multiply all pair values.
    int worker_count = n;
    int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
    int num_blocks = (worker_count + num_threads - 1) / num_threads;

    mul_pair_kernel <<< num_blocks, num_threads >>> (table1, table2, device_tmp, n);

    // 2. Sum up all the values in the array.
    sum_single_array(device_tmp, n);
    cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyHostToHost);

    cudaFree(device_tmp);

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t bk_sum_all_case_2(
    S* arr, S* output, int n)
  {
    CHK_INIT_IF_RETURN();

    // Sum up all the values in the array.
    sum_single_array(arr, n);
    cudaMemcpy(output, arr, sizeof(S), cudaMemcpyHostToHost);

    return CHK_LAST();
  }

  /////////////////////////////////
  /// BookKeeping produce
  /////////////////////////////////

  // template <typename S>
  // cudaError_t bk_produce_case_1(
  //   S* table1, S* table2, S* output, int n)
  // {
  //   CHK_INIT_IF_RETURN();

  //   int cuda_device_ix = 0;
  //   cudaDeviceProp prop;
  //   cudaGetDeviceProperties(&prop, cuda_device_ix);

  //   S* device_tmp;
  //   // allocate device array
  //   cudaMalloc((void**)&device_tmp, n * sizeof(S));

  //   // 1. Multiply all pair values.
  //   int worker_count = n;
  //   int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
  //   int num_blocks = (worker_count + num_threads - 1) / num_threads;

  //   mul_pair_kernel <<< num_blocks, num_threads >>> (arr1, arr2, device_tmp, n);

  //   // 2. Sum up all the values in the array.
  //   sum_single_array(device_tmp, n);
  //   cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyHostToHost);

  //   cudaFree(device_tmp);

  //   return CHK_LAST();
  // }

}
