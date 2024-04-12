#include "circuit.cuh"
#include "common.cuh"

namespace virgo {
  template <typename S>
  __global__ void fold_multi_kernel(
    S inverse_two,
    S* domain,
    uint32_t domain_size,
    S random_point,
    S** evaluations,
    uint32_t evaluation_size,
    S** output)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t domain_step = domain_size / evaluation_size;

    uint32_t replica_index = tid / (evaluation_size >> 1);
    uint32_t output_index = tid % (evaluation_size >> 1);
    uint32_t evaluation_positive_index = output_index;
    uint32_t evaluation_negative_index = output_index + evaluation_size / 2;

    uint32_t inverse_domain_element_index = evaluation_size - output_index;
    if (output_index == 0) { inverse_domain_element_index = 0; }

    S inverse_p = domain[inverse_domain_element_index * domain_step];

    S tmp0 =
      random_point * inverse_p *
      (evaluations[replica_index][evaluation_positive_index] - evaluations[replica_index][evaluation_negative_index]) *
      inv_r_mont2<S>;
    S tmp1 = tmp0 + evaluations[replica_index][evaluation_positive_index] +
             evaluations[replica_index][evaluation_negative_index];
    output[replica_index][output_index] = inverse_two * tmp1;
  }

  template <typename S>
  cudaError_t fold_multi(
    S* domain,
    uint32_t domain_size,
    uint32_t num_replicas,
    S random_point,
    S** evaluations,
    uint32_t evaluation_size,
    S** output)
  {
    CHK_INIT_IF_RETURN();

    S inverse_two = S::inverse(S::from(2));

    auto [num_blocks, num_threads] = find_thread_block(num_replicas * (evaluation_size >> 1));
    fold_multi_kernel<<<num_blocks, num_threads>>>(
      inverse_two, domain, domain_size, random_point, evaluations, evaluation_size, output);

    return CHK_LAST();
  }
} // namespace virgo
