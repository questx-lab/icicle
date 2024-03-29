#pragma once
#ifndef __CIRCUIT_H
#define __CIRCUIT_H

#include <cstdint>
#include <iostream>

#include "curves/curve_config.cuh"
#include "utils/error_handler.cuh"
#include "utils/device_context.cuh"

#include "utils/utils.h"

namespace virgo {
  template <typename S>
  struct SparseMultilinearExtension {
    uint32_t size;

    uint32_t z_num_vars;
    uint32_t x_num_vars;
    uint32_t y_num_vars;

    // mle(z[i], x[i], y[i]) = evaluations[i].
    uint32_t* point_z;
    uint32_t* point_x;
    uint32_t* point_y;
    S* evaluations;

    uint8_t* z_indices_size;
    uint32_t** z_indices; // z_indexes[i][.] are the indices of z==i in point_z.

    uint8_t* x_indices_size;
    uint32_t** x_indices; // x_indexes[i][.] are the indices of x==i in point_x.

    uint8_t* y_indices_size;
    uint32_t** y_indices; // y_indexes[i][.] are the indices of y==i in point_y.
  };

  template <typename S>
  struct ReverseSparseMultilinearExtension {
    uint32_t size;

    uint32_t subset_num_vars;
    uint32_t real_num_vars;

    // mle(z[i], x[i]) = evaluations[i].
    uint32_t* point_subset;
    uint32_t* point_real;
    S* evaluations;

    // Reverse extension is a mapping of subset_index to real_index.
    // It exists exact ONE z and ONE x in the all mle points. So we don't need
    // to design this mle as the same as SparseMLE.
    uint32_t* subset_position; // z_index[i] are the index of z==i in point_z.
    uint32_t* real_position;   // x_index[i] are the index of x==i in point_x.
  };

  template <typename S>
  struct Layer {
    // These two attributes are used to compute size of extensions.
    uint8_t layer_index;
    uint8_t num_layers;
    uint32_t size;

    SparseMultilinearExtension<S>* constant_ext;
    SparseMultilinearExtension<S>* mul_ext;
    SparseMultilinearExtension<S>* forward_x_ext;
    SparseMultilinearExtension<S>* forward_y_ext;
  };

  template <typename S>
  struct Circuit {
    uint8_t num_layers;
    Layer<S>* layers;

    // reverse_ext[target_layer_index][source_layer_index]
    // Mapping subset_index - real_index
    ReverseSparseMultilinearExtension<S>** reverse_ext;
  };

} // namespace virgo

#endif
