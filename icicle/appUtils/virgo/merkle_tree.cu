#include "virgo.cuh"
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>
#include <tuple>

namespace virgo {

  const uint32_t D[8] = {1, 31, 19, 23, 13, 17, 7, 11};

  template <typename S>
  __device__ S hash_one_field(S x) {
    auto result = x * x;
    result = result * x;

    return result;
  }

  template <typename S>
  __global__ void mimc_hash_array(const MerkleTreeConfig<S>& config, S* arr, int n) {
    // auto num_repetitions = config.MAX_MIMC_K / n;
    // if (num_repetitions < 2) {
    //   num_repetitions = 2;
    // }

    // auto r = S::from(0);
    // for (int repetition_index = 0; repetition_index < repetition_index < num_repetitions; repetition_index++) {
    //   auto d = D[repetition_index % 8];
    //   auto start = repetition_index % n;
    //   for (int i = 0; i < n; i++) {
    //     auto k_index = (repetition_index * n + i) % config.MAX_MIMC_K;
    //     auto v_index = (start + d * i) % n;

    //     // r = hash_one_field()
    //   }
    // }
  }

  template <typename S>
  cudaError_t build_merkle_tree(const MerkleTreeConfig<S>& config, S* arr, S* output, int n) {
    printf("HEERRRRRR \n");
  }
}
