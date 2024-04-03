#include "virgo.cuh"
#include "sumcheck.cu"
#include "merkle_tree.cu"
#include "circuit.cu"
#include "gkr.cu"

namespace virgo {
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkSumAllCase1)(
    const SumcheckConfig& config,
    curve_config::scalar_t* arr1,
    curve_config::scalar_t* arr2,
    curve_config::scalar_t* output,
    int n)
  {
    return bk_sum_all_case_1<curve_config::scalar_t>(config, arr1, arr2, output, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkSumAllCase2)(
    const SumcheckConfig& config, curve_config::scalar_t* arr, curve_config::scalar_t* output, int n)
  {
    return bk_sum_all_case_2<curve_config::scalar_t>(config, arr, output, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkProduceCase1)(
    const SumcheckConfig& config,
    curve_config::scalar_t* table1,
    curve_config::scalar_t* table2,
    curve_config::scalar_t* output,
    int n)
  {
    return bk_produce_case_1<curve_config::scalar_t>(config, table1, table2, output, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkProduceCase2)(
    const SumcheckConfig& config, curve_config::scalar_t* table, curve_config::scalar_t* output, int n)
  {
    return bk_produce_case_2<curve_config::scalar_t>(config, table, output, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BkReduce)(
    const SumcheckConfig& config, curve_config::scalar_t* arr, int n, curve_config::scalar_t r)
  {
    return bk_reduce<curve_config::scalar_t>(config, arr, n, r);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, BuildMerkleTree)(
    const MerkleTreeConfig<curve_config::scalar_t>& config, curve_config::scalar_t* tree, int n)
  {
    return build_merkle_tree<curve_config::scalar_t>(config, tree, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, HashMerkleTreeSlice)(
    const MerkleTreeConfig<curve_config::scalar_t>& config,
    curve_config::scalar_t* input,
    curve_config::scalar_t* output,
    int n,
    int slice_size)
  {
    return hash_merkle_tree_slice<curve_config::scalar_t>(config, input, output, n, slice_size);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, CircuitEvaluate)(
    const virgo::Circuit<curve_config::scalar_t>& circuit,
    uint32_t num_subcircuits,
    curve_config::scalar_t** evaluations)
  {
    return circuit_evaluate<curve_config::scalar_t>(circuit, num_subcircuits, evaluations);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, CircuitSubsetEvaluations)(
    const virgo::Circuit<curve_config::scalar_t>& circuit,
    uint32_t num_subcircuits,
    uint8_t layer_index,
    curve_config::scalar_t** evaluations,
    curve_config::scalar_t** subset_evaluations)
  {
    return circuit_subset_evaluations<curve_config::scalar_t>(
      circuit, num_subcircuits, layer_index, evaluations, subset_evaluations);
  }

  extern "C" cudaError_t
  CONCAT_EXPAND(CURVE, MulByScalar)(curve_config::scalar_t* arr, curve_config::scalar_t scalar, uint32_t n)
  {
    return mul_by_scalar(arr, scalar, n);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, PrecomputeBookeeping)(
    curve_config::scalar_t init, curve_config::scalar_t* g, uint8_t g_size, curve_config::scalar_t* output)
  {
    return precompute_bookeeping(init, g, g_size, output);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, InitializePhase1Plus)(
    uint32_t num_replicas,
    uint32_t num_layers,
    uint32_t output_size,
    SparseMultilinearExtension<curve_config::scalar_t>* f_extensions,
    curve_config::scalar_t** s_evaluations,
    curve_config::scalar_t* bookeeping_g,
    curve_config::scalar_t* output)
  {
    return initialize_phase_1_plus(
      num_replicas, num_layers, output_size, f_extensions, s_evaluations, bookeeping_g, output);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, InitializePhase2Plus)(
    uint32_t num_replicas,
    uint32_t num_layers,
    uint32_t* on_host_output_size,
    SparseMultilinearExtension<curve_config::scalar_t>* f_extensions,
    curve_config::scalar_t* bookeeping_g,
    curve_config::scalar_t* bookeeping_u,
    curve_config::scalar_t** output)
  {
    return initialize_phase_2_plus(
      num_replicas, num_layers, on_host_output_size, f_extensions, bookeeping_g, bookeeping_u, output);
  }
} // namespace virgo
