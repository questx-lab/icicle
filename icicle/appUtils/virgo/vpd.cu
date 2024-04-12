#include "virgo.cuh"
#include "common.cuh"
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>
#include <tuple>

namespace virgo {

  template <typename S>
  __global__ void mul_by_scalar_multi_kernel(uint32_t num_arr, S** arr, S scalar)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t arr_index = tid % num_arr;
    uint32_t output_index = tid / num_arr;

    arr[arr_index][output_index] = arr[arr_index][output_index] * scalar * inv_r_mont<S>;
  }

  template <typename S>
  cudaError_t mul_by_scalar_multi(uint32_t num_arr, uint32_t size, S** arr, S scalar)
  {
    auto [num_blocks, num_threads] = find_thread_block(num_arr * size);
    mul_by_scalar_multi_kernel<<<num_blocks, num_threads>>>(num_arr, arr, scalar);

    return CHK_LAST();
  }

  template <typename S>
  __global__ void sub_arr_multi_kernel(uint32_t num_arr, S** a, S** b)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t arr_index = tid % num_arr;
    uint32_t index = tid / num_arr;

    a[arr_index][index] = a[arr_index][index] - b[arr_index][index];
  }

  template <typename S>
  cudaError_t sub_arr_multi(uint32_t num_arr, uint32_t size, S** a, S** b)
  {
    auto [num_blocks, num_threads] = find_thread_block(num_arr * size);
    sub_arr_multi_kernel<<<num_blocks, num_threads>>>(num_arr, a, b);

    return CHK_LAST();
  }

  template <typename S>
  __global__ void mul_arr_multi_kernel(uint32_t num_arr, S** a, S** b)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t arr_index = tid % num_arr;
    uint32_t index = tid / num_arr;

    a[arr_index][index] = a[arr_index][index] * b[arr_index][index] * inv_r_mont<S>;
  }

  template <typename S>
  cudaError_t mul_arr_multi(uint32_t num_arr, uint32_t size, S** a, S** b)
  {
    auto [num_blocks, num_threads] = find_thread_block(num_arr * size);
    mul_arr_multi_kernel<<<num_blocks, num_threads>>>(num_arr, a, b);

    return CHK_LAST();
  }

  template <typename S>
  __global__ void dense_mle_multi_kernel(uint32_t num_mle, S** arr, S** output, S r)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t mle_index = tid % num_mle;
    uint32_t output_index = tid / num_mle;

    uint32_t index = output_index * 2;

    output[mle_index][output_index] =
      arr[mle_index][index] + r * (arr[mle_index][index + 1] - arr[mle_index][index]) * inv_r_mont<S>;
  }

  template <typename S>
  __global__ void dense_mle_copy_intermediate(uint32_t num_mle, S** output, S** a)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t mle_index = tid % num_mle;
    uint32_t output_index = tid / num_mle;

    output[mle_index][output_index] = a[mle_index][output_index];
  }

  template <typename S>
  __global__ void dense_mle_copy_output(S* output, S** a)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid] = a[tid][0];
  }

  template <typename S>
  cudaError_t dense_mle_multi(uint32_t num_mle, S* output, S** arr, int evaluation_size, S* on_host_input)
  {
    // allocate device array
    S** host_tmp = (S**)malloc(num_mle * sizeof(S*));
    for (uint32_t i = 0; i < num_mle; i++) {
      CHK_IF_RETURN(cudaMalloc((void**)&host_tmp[i], (evaluation_size / 2) * sizeof(S)));
    }

    S** origin_device_tmp;
    S** device_tmp;
    CHK_IF_RETURN(cudaMalloc((void**)&device_tmp, num_mle * sizeof(S*)));
    CHK_IF_RETURN(cudaMemcpy(device_tmp, host_tmp, num_mle * sizeof(S*), cudaMemcpyHostToDevice));

    origin_device_tmp = device_tmp;

    uint32_t num_vars = log2(evaluation_size);
    for (uint32_t i = 0; i < num_vars; i++) {
      uint32_t output_size = evaluation_size / (1 << (i + 1));
      auto [num_blocks, num_threads] = find_thread_block(num_mle * output_size);
      dense_mle_multi_kernel<<<num_blocks, num_threads>>>(num_mle, arr, device_tmp, on_host_input[num_vars - i - 1]);

      // copy the output back to the original array.
      dense_mle_copy_intermediate<<<num_blocks, num_threads>>>(num_mle, arr, device_tmp);
    }

    auto [num_blocks, num_threads] = find_thread_block(num_mle);
    dense_mle_copy_output<<<num_blocks, num_threads>>>(output, arr);

    // free the tmp array.
    CHK_IF_RETURN(cudaFree(origin_device_tmp));

    for (uint32_t i = 0; i < num_mle; i++) {
      CHK_IF_RETURN(cudaFree(host_tmp[i]));
    }
    free(host_tmp);

    return CHK_LAST();
  }

} // namespace virgo
