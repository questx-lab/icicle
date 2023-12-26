#define CURVE_ID 1

#include <chrono>
#include <iostream>
#include <vector>

#include "../curves/curve_config.cuh"
#include "../primitives/field.cuh"
#include "../primitives/projective.cuh"
#include "cuda_utils.cuh"
#include "device_context.cuh"

#include "mont.cu"

typedef curve_config::scalar_t test_scalar;
typedef curve_config::projective_t test_projective;
typedef curve_config::affine_t test_affine;

int main()
{
  int N = 1000;

  test_scalar* scalars = new test_scalar[N];
  test_affine* points = new test_affine[N];

  test_scalar::RandHostMany(scalars, N);
  test_projective::RandHostManyAffine(points, N);
  std::cout << "finished generating" << std::endl;

  std::cout << "First point: " << points[0].x << " " << points[0].y << std::endl;

  test_scalar* scalars_d;
  test_affine* points_d;
  cudaMalloc(&scalars_d, sizeof(test_scalar) * N);
  cudaMalloc(&points_d, sizeof(test_affine) * N);
  cudaMemcpy(scalars_d, scalars, sizeof(test_scalar) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d, points, sizeof(test_affine) * N, cudaMemcpyHostToDevice);
  std::cout << "finished copying" << std::endl;

  device_context::DeviceContext ctx = device_context::get_default_device_context();
  mont::AffineConvertMontgomery(points_d, N, 1, ctx);
  std::cout << "finished converting to montgomery" << std::endl;

  cudaMemcpy(points, points_d, sizeof(test_affine) * N, cudaMemcpyDeviceToHost);
  std::cout << "First point: " << points[0].x << " " << points[0].y << std::endl;

  mont::AffineConvertMontgomery(points_d, N, 0, ctx);
  std::cout << "finished converting from montgomery" << std::endl;

  cudaMemcpy(points, points_d, sizeof(test_affine) * N, cudaMemcpyDeviceToHost);
  std::cout << "First point: " << points[0].x << " " << points[0].y << std::endl;
}