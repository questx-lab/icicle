#include "virgo.cuh"
#include "common.cuh"
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>
#include <tuple>

namespace virgo {
  /////////////////////////////////
  /// COMMON FUNCTIONS
  /////////////////////////////////
  template <typename S>
  __global__ void mul_pair_kernel(S* arr1, S* arr2, S* result, int n)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    result[tid] = arr1[tid] * arr2[tid] * (inv_r_mont<S>);
  }

  template <typename S>
  __global__ void reduce_sum_kernel(S* result, uint32_t n, uint32_t half)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto other = tid + half;
    if (other < n) { result[tid] = result[tid] + result[other]; }
  }

  template <typename S>
  __global__ void reduce_sum_kernel2(S* result, uint32_t m, uint32_t n, uint32_t half, uint32_t offset)
  {
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
    auto x = n;
    while (x > 1) {
      auto [num_blocks, num_threads] = find_thread_block((x * m) >> 1);

      int half = (x + 1) >> 1;
      reduce_sum_kernel2<<<num_blocks, num_threads>>>(arrs, m, x, half, n);

      x = (x + 1) >> 1;
    }
  }

  template <typename S>
  cudaError_t sum_single_array(S* arr, int n)
  {
    auto x = n;
    while (x > 1) {
      auto [num_blocks, num_threads] = find_thread_block(x >> 1);

      int half = (x + 1) >> 1;
      reduce_sum_kernel<<<num_blocks, num_threads>>>(arr, x, half);

      x = (x + 1) >> 1;
    }
  }

  /////////////////////////////////
  /// BookKeeping sum_all
  /////////////////////////////////

  template <typename S>
  cudaError_t bk_sum_all_case_1(const SumcheckConfig& config, S* table1, S* table2, S* output, int n)
  {
    CHK_INIT_IF_RETURN();

    S* device_tmp;
    // allocate device array
    CHK_IF_RETURN(cudaMalloc((void**)&device_tmp, n * sizeof(S)));

    auto [num_blocks, num_threads] = find_thread_block(n);
    mul_pair_kernel<<<num_blocks, num_threads>>>(table1, table2, device_tmp, n);

    // 2. Sum up all the values in the array.
    sum_single_array(device_tmp, n);
    CHK_IF_RETURN(cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyDeviceToDevice));

    CHK_IF_RETURN(cudaFree(device_tmp));

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t bk_sum_all_case_2(const SumcheckConfig& config, S* arr, S* output, int n)
  {
    CHK_INIT_IF_RETURN();

    S* device_tmp;
    // allocate device array
    CHK_IF_RETURN(cudaMalloc((void**)&device_tmp, n * sizeof(S)));
    CHK_IF_RETURN(cudaMemcpy(device_tmp, arr, n * sizeof(S), cudaMemcpyDeviceToDevice));

    // Sum up all the values in the array.
    sum_single_array(device_tmp, n);
    // copy the result to output
    CHK_IF_RETURN(cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyDeviceToDevice));

    // free the temp array.
    CHK_IF_RETURN(cudaFree(device_tmp));

    return CHK_LAST();
  }

  /////////////////////////////////
  /// BookKeeping produce
  /////////////////////////////////

  template <typename S>
  __global__ void bk_produce_case_1_multiply(S* table1, S* table2, S* output, int n)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto half_n = n >> 1;
    if (tid >= half_n * 3) { return; }

    S result;
    // auto pid = tid / half_n;
    auto pid = tid / half_n;
    auto output_index = tid % half_n;
    auto i0 = output_index * 2;
    auto i1 = i0 + 1;

    if (pid == 0) {
      // result.0 += *a10 * a20;
      output[output_index] = table1[i0] * table2[i0] * inv_r_mont<S>;
    } else if (pid == 1) {
      // result.1 += *a11 * a21;
      output[n / 2 + output_index] = table1[i1] * table2[i1] * inv_r_mont<S>;
    } else {
      // result.2 += (*a11 + a11 - a10) * (*a21 + a21 - a20);
      S two = S::from(2);
      output[n + output_index] = (two * table1[i1] - table1[i0]) * (two * table2[i1] - table2[i0]) * inv_r_mont<S>;
    }
  }

  template <typename S>
  cudaError_t bk_produce_case_1(const SumcheckConfig& config, S* table1, S* table2, S* output, int n)
  {
    CHK_INIT_IF_RETURN();

    auto half_n = n / 2;
    auto sum_len = 3 * half_n;

    S* device_tmp;
    // allocate device array
    CHK_IF_RETURN(cudaMalloc((void**)&device_tmp, sum_len * sizeof(S)));

    // Step 1. Multiply
    auto [num_blocks, num_threads] = find_thread_block(sum_len);
    bk_produce_case_1_multiply<<<num_blocks, num_threads>>>(table1, table2, device_tmp, n);

    auto err2 = CHK_LAST();

    // Step 2. Sum up.
    sum_arrays(device_tmp, 3, half_n);

    CHK_IF_RETURN(cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyDeviceToDevice));
    CHK_IF_RETURN(cudaMemcpy(output + 1, device_tmp + half_n, sizeof(S), cudaMemcpyDeviceToDevice));
    CHK_IF_RETURN(cudaMemcpy(output + 2, device_tmp + n, sizeof(S), cudaMemcpyDeviceToDevice));

    CHK_IF_RETURN(cudaFree(device_tmp));

    return CHK_LAST();
  }

  template <typename S>
  __global__ void bk_produce_case_2_multiply(S* table, S* output, int n)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto half_n = n >> 1;
    if (tid >= half_n * 3) { return; }

    S result;
    // auto pid = tid / half_n;
    auto pid = tid / half_n;
    auto output_index = tid % half_n;
    auto i0 = output_index * 2;
    auto i1 = i0 + 1;

    if (pid == 0) {
      // result.0 += a10;
      output[output_index] = table[i0];
    } else if (pid == 1) {
      // result.1 += a11;
      output[n / 2 + output_index] = table[i1];
    } else {
      // result.2 += *a11 + a11 - a10;
      S two = S::from(2);
      output[n + output_index] = two * table[i1] - table[i0];
    }
  }

  template <typename S>
  cudaError_t bk_produce_case_2(const SumcheckConfig& config, S* table, S* output, int n)
  {
    CHK_INIT_IF_RETURN();

    auto half_n = n / 2;
    auto sum_len = 3 * half_n;

    S* device_tmp;
    // allocate device array
    CHK_IF_RETURN(cudaMalloc((void**)&device_tmp, sum_len * sizeof(S)));

    // Step 1. Multiply
    auto [num_blocks, num_threads] = find_thread_block(sum_len);
    bk_produce_case_2_multiply<<<num_blocks, num_threads>>>(table, device_tmp, n);

    auto err2 = CHK_LAST();

    // Step 2. Sum up.
    // sum_single_array(device_tmp, n / 2);
    sum_arrays(device_tmp, 3, half_n);

    CHK_IF_RETURN(cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyDeviceToDevice));
    CHK_IF_RETURN(cudaMemcpy(output + 1, device_tmp + half_n, sizeof(S), cudaMemcpyDeviceToDevice));
    CHK_IF_RETURN(cudaMemcpy(output + 2, device_tmp + n, sizeof(S), cudaMemcpyDeviceToDevice));

    CHK_IF_RETURN(cudaFree(device_tmp));

    return CHK_LAST();
  }

  template <typename S>
  __global__ void run_bk_reduce(S* arr, S* output, int n, S r)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto index = tid * 2;
    output[tid] = arr[index] + r * (arr[index + 1] - arr[index]);
  }

  template <typename S>
  cudaError_t bk_reduce(const SumcheckConfig& config, S* arr, int n, S r)
  {
    // allocate device array
    S* device_tmp;
    CHK_IF_RETURN(cudaMalloc((void**)&device_tmp, n / 2 * sizeof(S)));

    auto [num_blocks, num_threads] = find_thread_block(n / 2);
    run_bk_reduce<<<num_blocks, num_threads>>>(arr, device_tmp, n / 2, r);

    // copy the output back to the original array.
    CHK_IF_RETURN(cudaMemcpy(arr, device_tmp, n / 2 * sizeof(S), cudaMemcpyDeviceToDevice));

    // free the tmp array.
    CHK_IF_RETURN(cudaFree(device_tmp));

    return CHK_LAST();
  }

  template <typename S>
  __global__ void mul_by_scalar_kernel(S* arr, S scalar, uint32_t n)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) { return; }

    arr[tid] = arr[tid] * scalar * inv_r_mont<S>;
  }

  template <typename S>
  cudaError_t mul_by_scalar(S* arr, S scalar, uint32_t n)
  {
    auto [num_blocks, num_threads] = find_thread_block(n);
    mul_by_scalar_kernel<<<num_blocks, num_threads>>>(arr, scalar, n);

    return CHK_LAST();
  }
} // namespace virgo
