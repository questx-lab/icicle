#include "circuit.cuh"
#include "common.cuh"

namespace virgo {
  __device__ const uint8_t GATE_CONST = 0;
  __device__ const uint8_t GATE_MULTIPLICATION = 1;
  __device__ const uint8_t GATE_FORWARD_X = 2;
  __device__ const uint8_t GATE_FORWARD_Y = 3;

  template <typename S>
  __device__ S calculate_gate(uint8_t gate_type, S c, S in1, S in2)
  {
    if (gate_type == GATE_CONST) {
      return c;
    } else if (gate_type == GATE_MULTIPLICATION) {
      S result = c * in1 * in2;
      return result * inv_r_mont2<S>;
    } else if (gate_type == GATE_FORWARD_X) {
      S result = c * in1;
      return result * inv_r_mont<S>;
    } else if (gate_type == GATE_FORWARD_Y) {
      S result = c * in2;
      return result * inv_r_mont<S>;
    } else {
      panic();
    }
  }

  template <typename S>
  __global__ void evaluate_single_gate_type(
    uint8_t num_layers,
    uint8_t layer_index,
    uint32_t current_layer_size,
    uint8_t gate_type,
    SparseMultilinearExtension<S>* ext,
    ReverseSparseMultilinearExtension** reverse_exts,
    S** evaluations)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto z_index = tid;
    auto subcircuit_index = z_index / current_layer_size;
    auto relative_z_index = z_index % current_layer_size;

    for (uint8_t target_layer_index = layer_index + 1; target_layer_index < num_layers + 1; target_layer_index++) {
      uint8_t ext_index = target_layer_index - layer_index - 1;
      SparseMultilinearExtension<S> target_ext = ext[ext_index];
      uint32_t next_layer_size = 1 << target_ext.x_num_vars;

      uint32_t start = target_ext.z_indices_start[relative_z_index];
      uint32_t end = target_ext.z_indices_start[relative_z_index + 1];

      for (uint8_t i = start; i < end; i++) {
        uint32_t k = target_ext.z_indices[i];

        printf("type=%d k=%d i=%d expected=%d  got=%d\n", gate_type, k, i, target_ext.point_z[k], relative_z_index);

        if (target_ext.point_z[k] != relative_z_index) { panic(); }

        uint32_t y_layer_size = 1 << reverse_exts[target_layer_index][layer_index].real_num_vars;

        uint32_t relative_x_index = target_ext.point_x[k];
        uint32_t x_index = subcircuit_index * next_layer_size + relative_x_index;

        // Currently, y_index is a subset index, we must convert it to real index.
        uint32_t relative_y_subset_index = target_ext.point_y[k];
        uint32_t y_index_position =
          reverse_exts[target_layer_index][layer_index].subset_position[relative_y_subset_index];
        uint32_t relative_y_real_index = reverse_exts[target_layer_index][layer_index].point_real[y_index_position];
        uint32_t y_real_index = subcircuit_index * y_layer_size + relative_y_real_index;

        S c = target_ext.evaluations[k];
        S x = evaluations[layer_index + 1][x_index];
        S y = evaluations[target_layer_index][y_real_index];

        evaluations[layer_index][z_index] = evaluations[layer_index][z_index] + calculate_gate(gate_type, c, x, y);
      }
    }
  }

  template <typename S>
  cudaError_t layer_evaluate(
    uint32_t num_subcircuits,
    uint8_t num_layers,
    uint8_t layer_index,
    const Layer<S>& layer,
    ReverseSparseMultilinearExtension** reverse_exts,
    S** evaluations)
  {
    CHK_INIT_IF_RETURN();

    // We need to evaluate 2^num_vars gates of z.
    auto [num_blocks, num_threads] = find_thread_block(layer.size * num_subcircuits);

    evaluate_single_gate_type<<<num_blocks, num_threads>>>(
      num_layers, layer_index, layer.size, GATE_CONST, layer.constant_ext, reverse_exts, evaluations);

    evaluate_single_gate_type<<<num_blocks, num_threads>>>(
      num_layers, layer_index, layer.size, GATE_MULTIPLICATION, layer.mul_ext, reverse_exts, evaluations);

    evaluate_single_gate_type<<<num_blocks, num_threads>>>(
      num_layers, layer_index, layer.size, GATE_FORWARD_X, layer.forward_x_ext, reverse_exts, evaluations);

    evaluate_single_gate_type<<<num_blocks, num_threads>>>(
      num_layers, layer_index, layer.size, GATE_FORWARD_Y, layer.forward_y_ext, reverse_exts, evaluations);

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t circuit_evaluate(const Circuit<S>& circuit, uint32_t num_subcircuits, S** evaluations)
  {
    for (int8_t layer_index = circuit.num_layers - 1; layer_index >= 0; layer_index--) {
      printf("LAYERXXXXXXXXXXXXXXXXX %d\n", layer_index);
      layer_evaluate(
        num_subcircuits, circuit.num_layers, layer_index, circuit.layers[layer_index], circuit.reverse_exts,
        evaluations);

      if (layer_index == 2) { break; }
    }

    return CHK_LAST();
  }

  template <typename S>
  __global__ void extract_subset_evaluation(
    ReverseSparseMultilinearExtension** reverse_exts,
    uint8_t source_layer_index,
    uint8_t target_layer_index,
    S** evaluations,
    S** subset_evaluations)
  {
    ReverseSparseMultilinearExtension reverse_ext = reverse_exts[target_layer_index][source_layer_index];

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t subset_index = tid;
    uint32_t subcircuit_index = tid / (1 << reverse_ext.subset_num_vars);
    uint32_t relative_subset_index = tid % (1 << reverse_ext.subset_num_vars);

    uint32_t point_position = reverse_ext.subset_position[relative_subset_index];
    if (point_position == 4294967295) { return; }

    uint32_t relative_real_index = reverse_ext.point_real[point_position];
    uint32_t real_index = subcircuit_index * (1 << reverse_ext.real_num_vars) + relative_real_index;
    subset_evaluations[target_layer_index - source_layer_index - 1][subset_index] =
      evaluations[target_layer_index][real_index];
  }

  template <typename S>
  cudaError_t circuit_subset_evaluations(
    const Circuit<S>& circuit, uint32_t num_subcircuits, uint8_t layer_index, S** evaluations, S** subset_evaluations)
  {
    CHK_INIT_IF_RETURN();

    for (uint8_t target_layer_index = layer_index + 1; target_layer_index < circuit.num_layers + 1;
         target_layer_index++) {
      auto [num_blocks, num_threads] =
        find_thread_block(num_subcircuits * (1 << circuit.on_host_subset_num_vars[target_layer_index][layer_index]));

      extract_subset_evaluation<<<num_blocks, num_threads>>>(
        circuit.reverse_exts, layer_index, target_layer_index, evaluations, subset_evaluations);
    }

    return CHK_LAST();
  }
} // namespace virgo
