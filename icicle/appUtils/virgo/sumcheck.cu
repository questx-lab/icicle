#include "virgo.cuh"
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>
#include <tuple>

namespace virgo {
  // This is arkwork inverse R, not icicle inverse R.
  // inv_r = 9915499612839321149637521777990102151350674507940716049588462388200839649614
  // inv_r2 = inv_r ^ 2 = 8519677608991584271437967308266649112183478179623991153221810821821888926024
  template <typename S>
  __device__ constexpr S inv_r_mont = S({0x6db1194e, 0xdc5ba005, 0xe111ec87, 0x90ef5a9, 0xaeb85d5d, 0xc8260de4, 0x82c5551c, 0x15ebf951});

  template <typename S>
  __device__ constexpr S inv_r_mont2 = S({0xd3c71148, 0xae12ba81, 0xb38e2428, 0x52f28270, 0x79a1edeb, 0xe065f3e3, 0xe436631e, 0x12d5f775});

  // template <typename S>
  // __device__ constexpr S two = S({0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x00000002});

  /////////////////////////////////
  /// COMMON FUNCTIONS
  /////////////////////////////////

  std::tuple<int, int> find_thread_block(int n) {
    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);

    // std::cout << "prop.maxThreadsPerBlock = " << prop.maxThreadsPerBlock << std::endl;

    int worker_count = n;
    int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
    int num_blocks = (worker_count + num_threads - 1) / num_threads;

    return std::make_tuple(num_blocks, num_threads);
  }

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
    auto x = n;
    while (x > 1) {
      auto [num_blocks, num_threads] = find_thread_block((x * m) >> 1);

      int half = (x + 1) >> 1;
      reduce_sum_kernel2 <<< num_blocks, num_threads >>> (arrs, m, x, half, n);

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
      reduce_sum_kernel <<< num_blocks, num_threads >>> (arr, x, half);

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
    cudaMalloc((void**)&device_tmp, n * sizeof(S));

    auto [num_blocks, num_threads] = find_thread_block(n);
    mul_pair_kernel <<< num_blocks, num_threads >>> (table1, table2, device_tmp, n);

    // 2. Sum up all the values in the array.
    sum_single_array(device_tmp, n);
    cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyHostToHost);

    cudaFree(device_tmp);

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t bk_sum_all_case_2(const SumcheckConfig& config, S* arr, S* output, int n)
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

  template <typename S>
  __global__ void bk_produce_case_1_multiply(S* table1, S* table2, S* output, int n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto half_n = n >> 1;
    if (tid >= half_n * 3) {
      return;
    }

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
    auto start = std::chrono::high_resolution_clock::now();

    auto half_n = n / 2;
    auto sum_len = 3 * half_n;

    S* device_tmp;
    // allocate device array
    CHK_IF_RETURN(cudaMalloc((void**)&device_tmp, sum_len * sizeof(S)));

    // Step 1. Multiply
    auto [num_blocks, num_threads] = find_thread_block(sum_len);
    // If we set num_threads = 1024 (max thread), we would get "too many resources requested for launch"
    // https://stackoverflow.com/a/29901673
    // We work around this by reducing the number of thread per block and increasing num_blocks.
    if (num_threads == 1024) {
      num_threads /= 2;
      num_blocks *= 2;
    }
    bk_produce_case_1_multiply <<< num_blocks, num_threads >>> (table1, table2, device_tmp, n);

    auto err2 = CHK_LAST();

    // Step 2. Sum up.
    sum_arrays(device_tmp, 3, half_n);

    cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyHostToHost);
    cudaMemcpy(output + 1, device_tmp + half_n, sizeof(S), cudaMemcpyHostToHost);
    cudaMemcpy(output + 2, device_tmp + n, sizeof(S), cudaMemcpyHostToHost);

    cudaFree(device_tmp);

    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start);
    std::cout << "bk_produce_case_1, GPU Duration = " << duration1.count() << std::endl;

    return CHK_LAST();
  }

  template <typename S>
  __global__ void bk_produce_case_2_multiply(S* table, S* output, int n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto half_n = n >> 1;
    if (tid >= half_n * 3) {
      return;
    }

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
    auto start = std::chrono::high_resolution_clock::now();

    auto half_n = n / 2;
    auto sum_len = 3 * half_n;

    S* device_tmp;
    // allocate device array
    CHK_IF_RETURN(cudaMalloc((void**)&device_tmp, sum_len * sizeof(S)));

    // Step 1. Multiply
    auto [num_blocks, num_threads] = find_thread_block(sum_len);
    bk_produce_case_2_multiply <<< num_blocks, num_threads >>> (table, device_tmp, n);

    auto err2 = CHK_LAST();

    // Step 2. Sum up.
    // sum_single_array(device_tmp, n / 2);
    sum_arrays(device_tmp, 3, half_n);

    cudaMemcpy(output, device_tmp, sizeof(S), cudaMemcpyHostToHost);
    cudaMemcpy(output + 1, device_tmp + half_n, sizeof(S), cudaMemcpyHostToHost);
    cudaMemcpy(output + 2, device_tmp + n, sizeof(S), cudaMemcpyHostToHost);

    cudaFree(device_tmp);

    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start);
    std::cout << "bk_produce_case_2, GPU Duration (ms) = " << duration1.count() << std::endl;

    return CHK_LAST();
  }
}
