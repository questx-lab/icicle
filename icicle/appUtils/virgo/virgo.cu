#include "virgo.cuh"
#include "sumcheck.cu"
#include "merkle_tree.cu"

namespace virgo {
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkSumAllCase1)(
    const SumcheckConfig &config,
    curve_config::scalar_t* arr1,
    curve_config::scalar_t* arr2,
    curve_config::scalar_t* output,
    int n)
  {
    return bk_sum_all_case_1<curve_config::scalar_t>(config, arr1, arr2, output, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkSumAllCase2)(
    const SumcheckConfig &config,
    curve_config::scalar_t* arr,
    curve_config::scalar_t* output,
    int n)
  {
    return bk_sum_all_case_2<curve_config::scalar_t>(config, arr, output, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkProduceCase1)(
    const SumcheckConfig &config,
    curve_config::scalar_t* table1,
    curve_config::scalar_t* table2,
    curve_config::scalar_t* output,
    int n)
  {
    return bk_produce_case_1<curve_config::scalar_t>(config, table1, table2, output, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkProduceCase2)(
    const SumcheckConfig &config,
    curve_config::scalar_t* table,
    curve_config::scalar_t* output,
    int n)
  {
    return bk_produce_case_2<curve_config::scalar_t>(config, table, output, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BuildMerkleTree) (
    const MerkleTreeConfig<curve_config::scalar_t> &config,
    curve_config::scalar_t* tree,
    int n,
    int slice_size)
  {
    return build_merkle_tree<curve_config::scalar_t>(config, tree, n, slice_size);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, HashMerkleTreeSlice) (
    const MerkleTreeConfig<curve_config::scalar_t> &config,
    curve_config::scalar_t* input,
    curve_config::scalar_t* output,
    int n,
    int slice_size)
  {
    return hash_merkle_tree_slice<curve_config::scalar_t>(config, input, output, n, slice_size);
  }
}
