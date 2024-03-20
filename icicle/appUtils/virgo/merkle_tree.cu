#include "virgo.cuh"
#include "common.cuh"
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>
#include <tuple>

namespace virgo {

  // __global__ const uint32_t D[8] = {1, 31, 19, 23, 13, 17, 7, 11};

  template <typename S>
  __device__ S hash_one_field(S x) {
    auto result = x * x;
    result = result * x;

    return result * inv_r_mont2<S>;
  }

  template <typename S>
  __global__ void mimc_hash_array(const MerkleTreeConfig<S> config, S* arr, S* output, int n) {
    auto num_repetitions = config.max_mimc_k / n;
    if (num_repetitions < 2) {
      num_repetitions = 2;
    }

    auto r = S::from(0);
    for (int repetition_index = 0; repetition_index < num_repetitions; repetition_index++) {
    // for (int repetition_index = 0; repetition_index < 1; repetition_index++) {
      uint32_t d = config.D[repetition_index % 8];

      auto start = repetition_index % n;
      for (int i = 0; i < n; i++) {
      // for (int i = 0; i < 1; i++) {
        auto k_index = (repetition_index * n + i) % config.max_mimc_k;
        auto v_index = (start + d * i) % n;

        r = hash_one_field(r + arr[v_index] + config.mimc_params[k_index]);
      }
    }

    output[0] = r;
  }

  template <typename S>
  cudaError_t build_merkle_tree(const MerkleTreeConfig<S>& config, S* arr, S* output, int n) {
    print_arr(arr, 0, 1);
    print_arr(config.mimc_params, 0, 1);
    // print_u32_arr(config.D, 0, 1);

    // S* device_tmp;
    // // allocate device array
    // cudaMalloc((void**)&device_tmp, 1 * sizeof(S));

    auto [num_blocks, num_threads] = find_thread_block(n);
    mimc_hash_array <<< 1, 1 >>> (config, arr, output, n);

    print_arr(output, 0, 1);

    return CHK_LAST();
  }
}
