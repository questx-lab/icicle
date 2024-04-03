#include "circuit.cuh"
#include "common.cuh"

namespace virgo {
  template <typename S>
  __global__ void precompute_bookeeping_kernel(S* g, uint8_t g_index, S* prev_output, S* output)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t prev_output_size = 1 << g_index;
    uint32_t curr_output_size = prev_output_size * 2;

    if (tid >= curr_output_size) { return; };

    uint32_t prev_tid = tid >> 1;

    output[tid] = prev_output[prev_tid] * g[g_index] * inv_r_mont<S>;

    if (tid % 2 == 0) { output[tid] = prev_output[prev_tid] - output[tid]; }
  }

  template <typename S>
  cudaError_t precompute_bookeeping(S init, S* g, uint8_t g_size, S* output)
  {
    CHK_INIT_IF_RETURN();

    if (g_size > 0) {
      S* intermediate_output;

      CHK_IF_RETURN(cudaMalloc((void**)&intermediate_output, (1 << g_size) * sizeof(S)));
      CHK_IF_RETURN(cudaMemcpy(intermediate_output, &init, sizeof(S), cudaMemcpyHostToDevice));

      for (uint8_t i = 0; i < g_size; i++) {
        uint32_t curr_output_size = 1 << (i + 1);
        auto [num_blocks, num_threads] = find_thread_block(curr_output_size);
        precompute_bookeeping_kernel<<<num_blocks, num_threads>>>(g, i, intermediate_output, output);

        CHK_IF_RETURN(cudaMemcpy(intermediate_output, output, curr_output_size * sizeof(S), cudaMemcpyDeviceToDevice));
      }

      CHK_IF_RETURN(cudaFree(intermediate_output));
    } else {
      CHK_IF_RETURN(cudaMemcpy(output, &init, sizeof(S), cudaMemcpyHostToDevice));
    }

    return CHK_LAST();
  }

  template <typename S>
  __global__ void update_bookeeping_phase_1(
    uint8_t num_layers, SparseMultilinearExtension<S>* f_extensions, S** s_evaluations, S* bookeeping_g, S* output)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // exts are extensions from layer i to j. So x_num_vars is the same at all
    // extensions.
    uint32_t x_num_vars = f_extensions[0].x_num_vars;

    uint32_t x_index = tid;
    uint32_t replica_index = x_index / (1 << x_num_vars);
    uint32_t relative_x_index = x_index % (1 << x_num_vars);

    for (uint8_t relative_layer_index = 0; relative_layer_index < num_layers; relative_layer_index++) {
      SparseMultilinearExtension<S> target_ext = f_extensions[relative_layer_index];

      uint32_t start = target_ext.x_indices_start[relative_x_index];
      uint32_t end = target_ext.x_indices_start[relative_x_index + 1];

      for (uint32_t position_index = start; position_index < end; position_index++) {
        uint32_t position = target_ext.x_indices[position_index];

        if (target_ext.point_x[position] != relative_x_index) { panic(); }

        uint32_t z_index = target_ext.point_z[position] + (replica_index << target_ext.z_num_vars);
        uint32_t y_index = target_ext.point_y[position] + (replica_index << target_ext.y_num_vars);

        S evaluation = target_ext.evaluations[position];

        output[x_index] = output[x_index] + bookeeping_g[z_index] * s_evaluations[relative_layer_index][y_index] *
                                              evaluation * inv_r_mont2<S>;
      }
    }
  }

  template <typename S>
  cudaError_t initialize_phase_1_plus(
    uint32_t num_replicas,
    uint32_t num_layers,
    uint32_t output_size,
    SparseMultilinearExtension<S>* f_extensions,
    S** s_evaluations,
    S* bookeeping_g,
    S* output)
  {
    CHK_INIT_IF_RETURN();

    auto [num_blocks, num_threads] = find_thread_block(output_size);
    update_bookeeping_phase_1<<<num_blocks, num_threads>>>(
      num_layers, f_extensions, s_evaluations, bookeeping_g, output);

    return CHK_LAST();
  }

  template <typename S>
  __global__ void update_bookeeping_phase_2(
    uint8_t relative_layer_index,
    SparseMultilinearExtension<S>* f_extensions,
    S* bookeeping_g,
    S* bookeeping_u,
    S** output)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    SparseMultilinearExtension<S> target_ext = f_extensions[relative_layer_index];

    uint32_t y_index = tid;
    uint32_t replica_index = y_index / (1 << target_ext.y_num_vars);
    uint32_t relative_y_index = y_index % (1 << target_ext.y_num_vars);

    uint32_t start = target_ext.y_indices_start[relative_y_index];
    uint32_t end = target_ext.y_indices_start[relative_y_index + 1];

    for (uint32_t position_index = start; position_index < end; position_index++) {
      uint32_t position = target_ext.y_indices[position_index];

      if (target_ext.point_y[position] != relative_y_index) { panic(); }

      uint32_t z_index = target_ext.point_z[position] + (replica_index << target_ext.z_num_vars);
      uint32_t x_index = target_ext.point_x[position] + (replica_index << target_ext.x_num_vars);

      S evaluation = target_ext.evaluations[position];

      output[relative_layer_index][y_index] = output[relative_layer_index][y_index] + bookeeping_g[z_index] *
                                                                                        bookeeping_u[x_index] *
                                                                                        evaluation * inv_r_mont2<S>;
    }
  }

  template <typename S>
  cudaError_t initialize_phase_2_plus(
    uint32_t num_replicas,
    uint32_t num_layers,
    uint32_t* on_host_output_size,
    SparseMultilinearExtension<S>* f_extensions,
    S* bookeeping_g,
    S* bookeeping_u,
    S** output)
  {
    CHK_INIT_IF_RETURN();

    for (uint8_t relative_layer_index = 0; relative_layer_index < num_layers; relative_layer_index++) {
      auto [num_blocks, num_threads] = find_thread_block(on_host_output_size[relative_layer_index]);

      update_bookeeping_phase_2<<<num_blocks, num_threads>>>(
        relative_layer_index, f_extensions, bookeeping_g, bookeeping_u, output);
    }

    return CHK_LAST();
  }
} // namespace virgo
