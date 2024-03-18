#pragma once
#ifndef __VIRGO_H
#define __VIRGO_H

#include <cstdint>
#include <iostream>

#include "curves/curve_config.cuh"
#include "utils/error_handler.cuh"
#include "utils/device_context.cuh"

#include "utils/utils.h"

namespace virgo {
  struct VirgoConfig {
    device_context::DeviceContext ctx;
  };

  template <typename S>
  struct MerkleTreeConfig {
    uint32_t D[8];
    uint32_t MAX_MIMC_K;
    S* mimc_params;
  };
}

#endif
