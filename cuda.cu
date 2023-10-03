#include "qdbmp.h"
#include "stdio.h"
#include "util.cuh"
#include <cuda_device_runtime_api.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

__device__ int d_abs(int a) {
  return a > 0 ? a : -a;
}

__device__ void shift_pixel(UCHAR * r, UCHAR * g, UCHAR * b, int width, int height) {
  r[0] = r[6231];
  g[0] = g[6231];
  b[0] = b[6231];
}

__device__ void negative_pixel(UCHAR * r, UCHAR * g, UCHAR * b) {
  *r = 255 - *r;
  *g = 255 - *g;
  *b = 255 - *b;
}

#define span 1
#define divider 8

__device__ void energy_pixel(UCHAR * r, UCHAR * g, UCHAR * b, UCHAR * o_r, UCHAR * o_g, UCHAR * o_b, int width, int height) {


  o_r[0] = (
          d_abs(r[0] - r[span])
          + d_abs(r[0] - r[-span])
          + d_abs(r[0] - r[width * span])
          + d_abs(r[0] - r[-width * span])
          + d_abs(r[0] - r[width * span + span])
          + d_abs(r[0] - r[width * span - span])
          + d_abs(r[0] - r[-width * span + span])
          + d_abs(r[0] - r[-width * span - span])
          ) / divider;
  o_g[0] = (
          d_abs(g[0] - g[span])
          + d_abs(g[0] - g[-span])
          + d_abs(g[0] - g[width * span])
          + d_abs(g[0] - g[-width * span])
          + d_abs(g[0] - g[width * span + span])
          + d_abs(g[0] - g[width * span - span])
          + d_abs(g[0] - g[-width * span + span])
          + d_abs(g[0] - g[-width * span - span])
          ) / divider;
  o_b[0] = (
          d_abs(b[0] - b[span])
          + d_abs(b[0] - b[-span])
          + d_abs(b[0] - b[width * span])
          + d_abs(b[0] - b[-width * span])
          + d_abs(b[0] - b[width * span + span])
          + d_abs(b[0] - b[width * span - span])
          + d_abs(b[0] - b[-width * span + span])
          + d_abs(b[0] - b[-width * span - span])
          ) / divider;
  UCHAR gray = o_r[0] * 0.3 + o_g[0] * 0.59 + o_b[0] * 0.11;
  o_r[0] = gray;
  o_g[0] = gray;
  o_b[0] = gray;
}

__global__ void negative_kernel(UCHAR * r, UCHAR * g, UCHAR * b,
                                UCHAR * o_r, UCHAR * o_g, UCHAR * o_b, int width, int height, int * sig) {
  *sig = *sig + 1;
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  // if (x >= width - 1 || y >= height - 1 || x <= 1 || y <= 1) return;
  int index = x + y * width;
  // if (index > width * height - 1) return;
  // r[index] = 255 - r[index];
  // g[index] = 255 - g[index];
  // b[index] = 255 - b[index];
  energy_pixel(&r[index], &g[index], &b[index], &o_r[index], &o_g[index], &o_b[index], width, height);
}

extern "C"
void test() {
  printf("Cuda test start\n");
  cuda_hello<<<1,1>>>();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  printf("Cuda test done\n");
}

extern "C"
void negative(UCHAR * r, UCHAR * g, UCHAR * b, int width, int height) {
  int * sig;
  dim3 threads(32, 32, 1);
  dim3 blocks((width / 32), (height / 32), 1);
  unsigned int pixelCount = width * height;

  UCHAR * o_r;
  UCHAR * o_g;
  UCHAR * o_b;
  cudaMalloc((void**)&o_r, pixelCount * sizeof(UCHAR));
  cudaMalloc((void**)&o_g, pixelCount * sizeof(UCHAR));
  cudaMalloc((void**)&o_b, pixelCount * sizeof(UCHAR));
  cudaMemcpy(o_r, r, pixelCount * sizeof(UCHAR), cudaMemcpyDeviceToDevice);
  cudaMemcpy(o_g, g, pixelCount * sizeof(UCHAR), cudaMemcpyDeviceToDevice);
  cudaMemcpy(o_b, b, pixelCount * sizeof(UCHAR), cudaMemcpyDeviceToDevice);

  cudaMalloc(&sig, sizeof(int));
  printf("Negative kernal starting ...\n");
  negative_kernel<<<blocks, threads>>>(r, g, b, o_r, o_g, o_b, width, height, sig);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );


  cudaMemcpy(r, o_r, pixelCount * sizeof(UCHAR), cudaMemcpyDeviceToDevice);
  cudaMemcpy(g, o_g, pixelCount * sizeof(UCHAR), cudaMemcpyDeviceToDevice);
  cudaMemcpy(b, o_b, pixelCount * sizeof(UCHAR), cudaMemcpyDeviceToDevice);

  int h_sig = 8;
  cudaMemcpy(&h_sig, sig, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Negative kernal finished. %d\n", h_sig);
}
