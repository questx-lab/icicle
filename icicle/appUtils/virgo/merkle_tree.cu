#include "virgo.cuh"
#include "common.cuh"
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>
#include <tuple>

namespace virgo {
  template <typename S>
  __device__ S hash_one_field(S x) {
    auto result = x * x;
    result = result * x;

    return result * inv_r_mont2<S>;
  }

  /**
   * Runs only on device to hash an array of numbers and store the output in an output array.
   */
  template <typename S>
  __device__ void device_mimc_hash_array(uint32_t tid, const MerkleTreeConfig<S> config, S* arr,
      S* output, int n) {
    auto num_repetitions = config.max_mimc_k / n;
    if (num_repetitions < 2) {
      num_repetitions = 2;
    }

    auto r = S::from(0);
    for (int repetition_index = 0; repetition_index < num_repetitions; repetition_index++) {
      uint32_t d = config.D[repetition_index % 8];

      auto start = repetition_index % n;
      for (int i = 0; i < n; i++) {
        auto k_index = (repetition_index * n + i) % config.max_mimc_k;
        auto v_index = (start + d * i) % n;

        r = hash_one_field(r + arr[v_index] + config.mimc_params[k_index]);
      }
    }

    output[tid] = r;
  }

  template <typename S>
  __global__ void mimc_hash_array(const MerkleTreeConfig<S> config, S* arr, S* output, int n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    device_mimc_hash_array(tid, config, arr + (tid << 1), output, n);
  }

  template <typename S>
  cudaError_t build_merkle_tree_no_slice(const MerkleTreeConfig<S>& config, S* tree, int n) {
    auto stream = config.ctx.stream;

    auto x = n;
    auto offset = 0;
    while (x > 1) {
      auto [num_blocks, num_threads] = find_thread_block(x >> 1);
      mimc_hash_array <<< num_blocks, num_threads, 0, stream >>> (config, tree + offset, tree + offset + x, 2);
      offset += x;
      x = x / 2;
    }

    return CHK_LAST();
  }

  template <typename S>
  __global__ void hash_slice(const MerkleTreeConfig<S> config, S* arr, S* output, int n, int slice_size) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    device_mimc_hash_array(0, config, arr + tid * slice_size, output + tid, slice_size);
  }

  template <typename S>
  cudaError_t hash_merkle_tree_slice(const MerkleTreeConfig<S>& config, S* input, S* output, int n, int slice_size) {
    auto slice_count = n / slice_size;
    auto [num_blocks, num_threads] = find_thread_block(slice_count);
    hash_slice <<< num_blocks, num_threads >>> (config, input, output, n, slice_size);

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t build_merkle_tree(const MerkleTreeConfig<S>& config, S* tree, int n, int slice_size) {
    return build_merkle_tree_no_slice(config, tree, n);
  }
}
