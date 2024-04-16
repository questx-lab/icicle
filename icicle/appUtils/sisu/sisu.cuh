#pragma once
#ifndef __SISU_H
#define __SISU_H

#include <cstdint>
#include <iostream>

#include "curves/curve_config.cuh"
#include "utils/error_handler.cuh"
#include "utils/device_context.cuh"

#include "utils/utils.h"

namespace sisu {
  struct SumcheckConfig {
    device_context::DeviceContext ctx;
  };

  template <typename S>
  struct MerkleTreeConfig {
    device_context::DeviceContext ctx;
    uint32_t max_mimc_k;
    S* mimc_params;
    uint32_t* D;
  };
} // namespace sisu

#endif
