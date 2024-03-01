#include <iostream>
#include <iomanip>
#include <chrono>
#include <nvml.h>
#include <vector>
#include <thread>

#define CURVE_ID 1
#include "curves/curve_config.cuh"
#include "utils/device_context.cuh"
#include "utils/vec_ops.cu"

using namespace curve_config;

typedef scalar_t S;

void print(std::string tag, S* b, int n) {
  std::cout << "=================" << std::endl;
  std::cout << "Printing " << tag << std::endl;

  for (int i = 0; i < n; i++) {
    std::cout << b[i] << " ";
  }
  std::cout << std::endl << "=================" << std::endl;
}

uint32_t reverse_bits(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

void old_reverse_bit(S* b, uint n) {
  for (int i = 1, j = 0; i < n; i++)
  {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;

    if (i < j) {
      // swap(b[i], b[j]);
      S tmp = b[i];
      b[i] = b[j];
      b[j] = tmp;
    }
  }
}

S* precompute_w(uint n, S root, S root_inv, uint root_pw, bool invert) {
  S* ws = (S*)malloc((n - 1) * sizeof(S));
  uint index = 0;

  for (int len = 2; len <= n; len <<= 1)
  {
    S wlen = invert ? root_inv : root;
    for (int i = len; i < root_pw; i <<= 1)
      wlen = wlen * wlen;

    S w = S::from(1);
    for (int j = 0; j < len / 2; j++)
    {
      ws[index++] = w;
      w = w * wlen;
    }
  }

  std::cout << "precompute_w index = " << index << std::endl;

  return ws;
}

void fft_cpu(S* b, uint n, S root, S root_inv, uint root_pw, S* ws, S* ws_inv, bool invert) {
  const int log_n = log2(n);
  for (int i = 0; i < n; i++) {
    uint rev = reverse_bits(i);
    rev = rev >> (32 - log_n);

    if (i < rev) {
      // std::cout << "Swapping " << i << " " << rev << std::endl;
      S tmp = b[i];
      b[i] = b[rev];
      b[rev] = tmp;
    }
  }
  // print("CPU value b after bit reverse", b, n);

  int ws_index = 0;
  for (int len = 2; len <= n; len <<= 1)
  {
    for (int i = 0; i < n; i += len)
    {
      for (int j = 0; j < len / 2; j++)
      {
        S w;
        if (!invert) {
          w = ws[ws_index + j];
        } else {
          w = ws_inv[ws_index + j];
        }

        S u = b[i + j];
        S v = b[i + j + len / 2] * w;
        b[i + j] = u + v;
        b[i + j + len / 2] = u - v;
      }
    }

    ws_index += len / 2;
  }

  if (invert) {
    S inv_n = S::inverse(S::from(n));
    for (int i = 0; i < n; i++) {
      b[i] = b[i] * inv_n;
    }
  }

  // print(b, n);
}

////////////////////////////////////////////////////////////////////////
__device__ uint32_t device_reverse_bits(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

__global__ void swap_bits(S* b, uint n, uint log_n) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = tid * 2; i < tid * 2 + 2; i++) {
    uint rev = device_reverse_bits(i);
    rev = rev >> (32 - log_n);

    if (i < rev) {
      S tmp = b[i];
      b[i] = b[rev];
      b[rev] = tmp;
    }
  }
}

__global__ void fft_kernel(S* b, uint n, S inv_n, uint pow, uint ws_index, S* ws, S* ws_inv, bool invert) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    int len = 1 << pow;
    int len2 = len >> 1; // len2 = len / 2
    int q = tid >> (pow - 1);
    int i = q * len;
    int j = tid - q * len2;

    S w;
    if (!invert) {
      w = ws[ws_index + j];
    } else {
      w = ws_inv[ws_index + j];
    }

    S u = b[i + j];
    S v = b[i + j + len / 2] * w;
    b[i + j] = u + v;
    b[i + j + len / 2] = u - v;
}

__global__ void invert_result(S* b, S inv_n) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto x = tid << 1;
  b[x] = b[x] * inv_n;
  b[x + 1] = b[x + 1] * inv_n;
}

void print_runtime(std::string message, std::chrono::time_point< std::chrono::system_clock > start_time) {
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << message << duration.count() << " microseconds" << std::endl;
}

S* fft_gpu(S* host_b, uint n, S* device_ws, S* device_ws_inv, bool invert, cudaStream_t stream) {
  cudaError_t err;
  S* device_b;

  S inv_n = S::inverse(S::from(n));

  auto start_time = std::chrono::high_resolution_clock::now();

  // allocate device array
  cudaMalloc((void**)&device_b, n * sizeof(S));

  // copy from host to device
  err = cudaMemcpyAsync(device_b, host_b, n * sizeof(S), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return NULL;
  }

  print_runtime("Copy HOST to DEVICE time: ", start_time);
  start_time = std::chrono::high_resolution_clock::now();

  int cuda_device_ix = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, cuda_device_ix);

  // Set the grid and block dimensions
  int worker_count = n >> 1;
  int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
  int num_blocks = (worker_count + num_threads - 1) / num_threads;

  const int log_n = log2(n);
  // Swap bits
  swap_bits<<< num_blocks, num_threads, 0, stream  >>> (device_b, n, log_n);

  // main loop
  int ws_index = 0;
  for (int pow = 1; ; pow++) {
    int len = 1 << pow;
    if (len > n) {
      break;
    }

    fft_kernel<<< num_blocks, num_threads, 0, stream  >>> (device_b, n, inv_n, pow, ws_index, device_ws, device_ws_inv, invert);

    ws_index += len >> 1;
  }

  // If this is interpolatio, invert the result
  if (invert) {
    invert_result<<< num_blocks, num_threads, 0, stream  >>> (device_b, inv_n);
  }

  print_runtime("Kernel Running time: ", start_time);

  S* host_result = (S*)malloc(n * sizeof(S));

  start_time = std::chrono::high_resolution_clock::now();
  err = cudaMemcpyAsync(host_result, device_b, n * sizeof(S), cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from device to host - " << cudaGetErrorString(err) << std::endl;
    return NULL;
  }

  print_runtime("Copy DEVICE to HOST time: ", start_time);
  start_time = std::chrono::high_resolution_clock::now();

  // printf("GPU num_blocks = %d, num_threads = %d\n", num_blocks, num_threads);

  return host_result;
}

void gpu_allocate(int n, S* &device_ws, S* &device_ws_inv) {
  const int log_n = log2(n);
  const int root_pw = 1 << log_n;

  S root = S::omega(log_n);
  S root_inv = S::inverse(root);

  S* ws = precompute_w(n, root, root_inv, root_pw, false);
  S* ws_inv = precompute_w(n, root, root_inv, root_pw, true);

  cudaMalloc((void**)&device_ws, n * sizeof(S));
  cudaMalloc((void**)&device_ws_inv, n * sizeof(S));
  // copy from host to device
  auto err = cudaMemcpy(device_ws, ws, n * sizeof(S), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    // return NULL;
  }
  err = cudaMemcpy(device_ws_inv, ws_inv, n * sizeof(S), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    // return NULL;
  }
}

S* run_gpu(std::vector<int> a) {
  const int n = a.size();
  S* device_ws;
  S* device_ws_inv;
  S* a_field = (S*)malloc(n * sizeof(S));
  for (int i = 0; i < a.size(); i++) {
    a_field[i] = S::from(a[i]);
  }

  gpu_allocate(n, device_ws, device_ws_inv);

  std::vector<std::thread> threads(n);

  const int num_streams = 8;
  cudaStream_t streams[num_streams];
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // fft_gpu(a_field, n, device_ws, device_ws_inv, false, streams[0]);

  // // function call
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_streams; i++) {
    // auto eval_result = fft_gpu(a_field, n, device_ws, device_ws_inv, false, streams[i]);
    threads[i] = std::thread(fft_gpu, a_field, n, device_ws, device_ws_inv, false, streams[i]);
  }

  for (int i = 0; i < num_streams; i++) {
    // cudaStreamSynchronize(streams[i]);
    threads[i].join();
  }

  print_runtime("Total running time of all streams: ", start_time);
  start_time = std::chrono::high_resolution_clock::now();
  // auto interpolate_result = fft_gpu(eval_result, n, device_ws, device_ws_inv, true, streams[0]);

  // for (int i = 0; i < 8; i++) {
  //   std::cout << eval_result[i] << std::endl;
  // }

  // return interpolate_result;

  return NULL;
}

S* run_cpu(std::vector<int> a) {
  const int log_n = log2(a.size());
  const int root_pw = 1 << log_n;

  int n = a.size();
  S* a_field = (S*)malloc(n * sizeof(S));
  for (int i = 0; i < a.size(); i++) {
    a_field[i] = S::from(a[i]);
  }

  S root = S::omega(log_n);
  S root_inv = S::inverse(root);

  // (uint n, S root, S root_inv, uint root_pw, bool invert)
  S* ws = precompute_w(n, root, root_inv, root_pw, false);
  S* ws_inv = precompute_w(n, root, root_inv, root_pw, true);

  std::cout << "Done precompute" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // Evaluate
  fft_cpu(a_field, n, root, root_inv, root_pw, ws, ws_inv, false);
  // print("CPU Evaluation", a_field, n);

  auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start);
  std::cout << "CPU Duration 1 = " << duration1.count() << std::endl;

  // Interpolate
  fft_cpu(a_field, n, root, root_inv, root_pw, ws, ws_inv, true);

  auto end2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1);
  std::cout << "CPU Duration 2 = " << duration2.count() << std::endl;

  return a_field;
}

void run_gpu_multiple() {
  // const int n = a.size();
  // S* device_ws;
  // S* device_ws_inv;
  // S* a_field = (S*)malloc(n * sizeof(S));
  // for (int i = 0; i < a.size(); i++) {
  //   a_field[i] = S::from(a[i]);
  // }

  // gpu_allocate(n, device_ws, device_ws_inv);
}

std::vector<int> gen_data() {
  // std::vector<int> a = {3, 1, 4, 1, 5, 9, 2, 6};
  std::vector<int> a;
  for (int i = 0; i < 1 << 20; i++) {
    int random = rand() % 1000;
    a.push_back(random);
  }

  return a;
}

int main(int argc, char** argv) {
  auto a = gen_data();

  // auto cpu_result = run_cpu(a);
  auto gpu_result = run_gpu(a);

  // for (int i = 0; i < a.size(); i++) {
  //   if (cpu_result[i] != gpu_result[i]) {
  //     std::cout << "Test fails at " << i << " cpu = " << cpu_result[i] << " gpu = " << gpu_result[i];
  //     std::cout << " original value = " << a[i] << std::endl;
  //     return 1;
  //   }
  // }

  // std::cout << "Test passed!" << std::endl;

  return 0;
}
