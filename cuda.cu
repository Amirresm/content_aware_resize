#include "qdbmp.h"
#include "stdio.h"
#include "util.cuh"
#include <cuda_device_runtime_api.h>

__device__ int d_abs(int a) { return a > 0 ? a : -a; }

#define span 1
#define divider 8

__device__ void energy_pixel(UCHAR *r, UCHAR *g, UCHAR *b, UCHAR *o_r,
                             UCHAR *o_g, UCHAR *o_b, int width, int height) {

  o_r[0] = (d_abs(r[0] - r[span]) + d_abs(r[0] - r[-span]) +
            d_abs(r[0] - r[width * span]) + d_abs(r[0] - r[-width * span]) +
            d_abs(r[0] - r[width * span + span]) +
            d_abs(r[0] - r[width * span - span]) +
            d_abs(r[0] - r[-width * span + span]) +
            d_abs(r[0] - r[-width * span - span])) /
           divider;
  o_g[0] = (d_abs(g[0] - g[span]) + d_abs(g[0] - g[-span]) +
            d_abs(g[0] - g[width * span]) + d_abs(g[0] - g[-width * span]) +
            d_abs(g[0] - g[width * span + span]) +
            d_abs(g[0] - g[width * span - span]) +
            d_abs(g[0] - g[-width * span + span]) +
            d_abs(g[0] - g[-width * span - span])) /
           divider;
  o_b[0] = (d_abs(b[0] - b[span]) + d_abs(b[0] - b[-span]) +
            d_abs(b[0] - b[width * span]) + d_abs(b[0] - b[-width * span]) +
            d_abs(b[0] - b[width * span + span]) +
            d_abs(b[0] - b[width * span - span]) +
            d_abs(b[0] - b[-width * span + span]) +
            d_abs(b[0] - b[-width * span - span])) /
           divider;
  UCHAR gray = o_r[0] * 0.3 + o_g[0] * 0.59 + o_b[0] * 0.11;
  o_r[0] = gray > 255 ? 255 : gray;
  o_g[0] = gray > 255 ? 255 : gray;
  o_b[0] = gray > 255 ? 255 : gray;
}

__global__ void energy_kernel(UCHAR *r, UCHAR *g, UCHAR *b, UCHAR *o_r,
                              UCHAR *o_g, UCHAR *o_b, int width, int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height) {
    return;
  }
  int index = x + y * width;
  energy_pixel(&r[index], &g[index], &b[index], &o_r[index], &o_g[index],
               &o_b[index], width, height);
}

extern "C" void energy(UCHAR *out_r, UCHAR *out_g, UCHAR *out_b, int width,
                       int height) {
  dim3 threads(32, 32);
  dim3 blocks(((width - 1) / 32) + 1, ((height - 1) / 32) + 1);
  unsigned int pixelCount = width * height;

  UCHAR *original_r;
  UCHAR *original_g;
  UCHAR *original_b;
  cudaMalloc((void **)&original_r, pixelCount * sizeof(UCHAR));
  cudaMalloc((void **)&original_g, pixelCount * sizeof(UCHAR));
  cudaMalloc((void **)&original_b, pixelCount * sizeof(UCHAR));

  cudaMemcpy(original_r, out_r, pixelCount * sizeof(UCHAR),
             cudaMemcpyHostToDevice);
  cudaMemcpy(original_g, out_g, pixelCount * sizeof(UCHAR),
             cudaMemcpyHostToDevice);
  cudaMemcpy(original_b, out_b, pixelCount * sizeof(UCHAR),
             cudaMemcpyHostToDevice);

  UCHAR *energy_r;
  UCHAR *energy_g;
  UCHAR *energy_b;
  cudaMalloc((void **)&energy_r, pixelCount * sizeof(UCHAR));
  cudaMalloc((void **)&energy_g, pixelCount * sizeof(UCHAR));
  cudaMalloc((void **)&energy_b, pixelCount * sizeof(UCHAR));

  energy_kernel<<<blocks, threads>>>(original_r, original_g, original_b,
                                     energy_r, energy_g, energy_b, width,
                                     height);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(out_r, energy_r, pixelCount * sizeof(UCHAR),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out_g, energy_g, pixelCount * sizeof(UCHAR),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out_b, energy_b, pixelCount * sizeof(UCHAR),
             cudaMemcpyDeviceToHost);

  cudaFree(original_r);
  cudaFree(original_g);
  cudaFree(original_b);
  cudaFree(energy_r);
  cudaFree(energy_g);
  cudaFree(energy_b);
}
