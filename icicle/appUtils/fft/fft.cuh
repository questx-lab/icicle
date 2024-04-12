#pragma once
#ifndef __FFT_H
#define __FFT_H

#include <cstdint>
#include <iostream>

#include "curves/curve_config.cuh"
#include "utils/error_handler.cuh"
#include "utils/device_context.cuh"
#include "utils/utils.h"

namespace fft {
  std::tuple<int, int> find_thread_block(int n)
  {
    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);

    int worker_count = n;
    int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
    int num_blocks = (worker_count + num_threads - 1) / num_threads;

    // If we set num_threads = 1024 (max thread), we would get "too many resources requested for launch"
    // https://stackoverflow.com/a/29901673
    // We work around this by reducing the number of thread per block and increasing num_blocks.
    if (num_threads == 1024) {
      num_threads /= 2;
      num_blocks *= 2;
    }

    return std::make_tuple(num_blocks, num_threads);
  }

  template <typename S>
  cudaError_t fft(S* input, S* output, S* ws, int n, bool invert);
} // namespace fft

#endif
