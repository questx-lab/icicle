#pragma once
#ifndef __VIRGO_COMMON_H
#define __VIRGO_COMMON_H

namespace virgo {
  // This is arkwork inverse R, not icicle inverse R.
  // inv_r = 9915499612839321149637521777990102151350674507940716049588462388200839649614
  // inv_r2 = inv_r ^ 2 = 8519677608991584271437967308266649112183478179623991153221810821821888926024
  template <typename S>
  __device__ constexpr S inv_r_mont =
    S({0x6db1194e, 0xdc5ba005, 0xe111ec87, 0x90ef5a9, 0xaeb85d5d, 0xc8260de4, 0x82c5551c, 0x15ebf951});

  template <typename S>
  __device__ constexpr S inv_r_mont2 =
    S({0xd3c71148, 0xae12ba81, 0xb38e2428, 0x52f28270, 0x79a1edeb, 0xe065f3e3, 0xe436631e, 0x12d5f775});

  __device__ void panic()
  {
    printf("cuda panic\n");
    int* a = 0;
    *a = 0;
  }

  template <typename S>
  void print_arr(S* arr, int start, int end)
  {
    int len = end - start;
    S* tmp = (S*)malloc(len * sizeof(S));

    cudaMemcpy(tmp, arr + start, len * sizeof(S), cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; i++) {
      std::cout << tmp[i] * inv_r_mont<S> << " ";
    }
    std::cout << std::endl;

    delete[] tmp;
  }

  void print_u32_arr(uint32_t* arr, int start, int end)
  {
    int len = end - start;
    uint32_t* tmp = (uint32_t*)malloc(len * sizeof(uint32_t));

    cudaMemcpy(tmp, arr + start, len * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; i++) {
      std::cout << tmp[i] << " ";
    }
    std::cout << std::endl;

    delete[] tmp;
  }

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
} // namespace virgo

#endif
