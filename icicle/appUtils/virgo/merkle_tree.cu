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

  template <typename S>
  __global__ void mimc_hash_array(const MerkleTreeConfig<S> config, S* arr, S* output, int n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto num_repetitions = config.max_mimc_k / n;
    if (num_repetitions < 2) {
      num_repetitions = 2;
    }

    auto offset = tid << 1;
    auto r = S::from(0);
    for (int repetition_index = 0; repetition_index < num_repetitions; repetition_index++) {
      uint32_t d = config.D[repetition_index % 8];

      auto start = repetition_index % n;
      for (int i = 0; i < n; i++) {
        auto k_index = (repetition_index * n + i) % config.max_mimc_k;
        auto v_index = (start + d * i) % n;

        r = hash_one_field(r + arr[offset + v_index] + config.mimc_params[k_index]);
      }
    }

    output[tid] = r;
  }

  template <typename S>
  cudaError_t build_merkle_tree(const MerkleTreeConfig<S>& config, S* tree, int n) {
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
}
