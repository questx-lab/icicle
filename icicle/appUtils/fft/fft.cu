#include "fft.cuh"

namespace fft {
  __device__ uint32_t device_reverse_bits(uint32_t x)
  {
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
  }

  template <typename S>
  __global__ void swap_bits(S* b, uint32_t n, uint32_t log_n)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid * 2; i < tid * 2 + 2; i++) {
      uint32_t rev = device_reverse_bits(i);
      rev = rev >> (32 - log_n);

      if (i < rev) {
        S tmp = b[i];
        b[i] = b[rev];
        b[rev] = tmp;
      }
    }
  }

  template <typename S>
  __global__ void fft_kernel(S* b, uint32_t n, uint32_t power, uint32_t ws_index, S* ws)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t len = 1 << power;
    uint32_t len2 = len >> 1; // len2 = len / 2
    uint32_t q = tid >> (power - 1);
    uint32_t i = q * len;
    uint32_t j = tid - q * len2;

    S w;
    w = ws[ws_index + j];

    S u = b[i + j];
    S v = b[i + j + len / 2] * w;
    b[i + j] = u + v;
    b[i + j + len / 2] = u - v;
  }

  template <typename S>
  __global__ void invert_result(S* b, S inv_n)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto x = tid << 1;
    b[x] = b[x] * inv_n;
    b[x + 1] = b[x + 1] * inv_n;
  }

  template <typename S>
  cudaError_t fft(S* device_inout, S* device_ws, int n, bool invert)
  {
    CHK_INIT_IF_RETURN();

    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);

    // Set the grid and block dimensions
    int worker_count = n >> 1;
    int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
    int num_blocks = (worker_count + num_threads - 1) / num_threads;

    const int log_n = log2(n);
    // Swap bits
    swap_bits<<<num_blocks, num_threads>>>(device_inout, n, log_n);

    // main loop
    int ws_index = 0;
    for (int pow = 1;; pow++) {
      int len = 1 << pow;
      if (len > n) { break; }

      fft_kernel<<<num_blocks, num_threads>>>(device_inout, n, pow, ws_index, device_ws);

      ws_index += len >> 1;
    }

    // If this is interpolation, invert the result
    if (invert) {
      S inv_n = S::inverse(S::from(n));
      invert_result<<<num_blocks, num_threads>>>(device_inout, inv_n);
    }

    return CHK_LAST();
  }

  template <typename S>
  __global__ void invert_result_multi(S** b, uint32_t n, S inv_n)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t fft_index = tid / (n / 2);
    uint32_t k = tid % (n / 2);

    auto x = k << 1;
    b[fft_index][x] = b[fft_index][x] * inv_n;
    b[fft_index][x + 1] = b[fft_index][x + 1] * inv_n;
  }

  template <typename S>
  __global__ void swap_bits_multi(S** b, uint32_t n, uint32_t log_n)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t z = tid / (n / 2);
    uint32_t k = tid % (n / 2);

    for (int i = k * 2; i < k * 2 + 2; i++) {
      uint32_t rev = device_reverse_bits(i);
      rev = rev >> (32 - log_n);

      if (i < rev) {
        S tmp = b[z][i];
        b[z][i] = b[z][rev];
        b[z][rev] = tmp;
      }
    }
  }

  template <typename S>
  __global__ void fft_multi_kernel(S** b, uint32_t n, uint32_t power, uint32_t ws_index, S* ws)
  {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t fft_index = tid / (n / 2);
    uint32_t k = tid % (n / 2);

    uint32_t len = 1 << power;
    uint32_t len2 = len >> 1; // len2 = len / 2
    uint32_t q = k >> (power - 1);
    uint32_t i = q * len;
    uint32_t j = k - q * len2;

    S w;
    w = ws[ws_index + j];

    S u = b[fft_index][i + j];
    S v = b[fft_index][i + j + len / 2] * w;
    b[fft_index][i + j] = u + v;
    b[fft_index][i + j + len / 2] = u - v;
  }

  template <typename S>
  cudaError_t fft_multi(int32_t num_fft, S** device_inout, S* device_ws, int n, bool invert)
  {
    CHK_INIT_IF_RETURN();

    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device_ix);

    // Set the grid and block dimensions
    auto [num_blocks, num_threads] = find_thread_block(num_fft * (n >> 1));

    const int log_n = log2(n);
    // Swap bits
    swap_bits_multi<<<num_blocks, num_threads>>>(device_inout, n, log_n);

    // main loop
    int ws_index = 0;
    for (int pow = 1;; pow++) {
      int len = 1 << pow;
      if (len > n) { break; }

      fft_multi_kernel<<<num_blocks, num_threads>>>(device_inout, n, pow, ws_index, device_ws);

      ws_index += len >> 1;
    }

    // If this is interpolation, invert the result
    if (invert) {
      S inv_n = S::inverse(S::from(n));
      invert_result_multi<<<num_blocks, num_threads>>>(device_inout, n, inv_n);
    }

    return CHK_LAST();
  }

  extern "C" cudaError_t
  CONCAT_EXPAND(CURVE, FftEvaluate)(curve_config::scalar_t* inout, curve_config::scalar_t* ws, int n)
  {
    return fft<curve_config::scalar_t>(inout, ws, n, false);
  }

  extern "C" cudaError_t
  CONCAT_EXPAND(CURVE, FftInterpolate)(curve_config::scalar_t* inout, curve_config::scalar_t* ws, int n)
  {
    return fft<curve_config::scalar_t>(inout, ws, n, true);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, FftEvaluateMulti)(
    uint32_t num_fft, curve_config::scalar_t** inout, curve_config::scalar_t* ws, int n)
  {
    return fft_multi(num_fft, inout, ws, n, false);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, FftInterpolateMulti)(
    uint32_t num_fft, curve_config::scalar_t** inout, curve_config::scalar_t* ws, int n)
  {
    return fft_multi(num_fft, inout, ws, n, true);
  }
} // namespace fft
