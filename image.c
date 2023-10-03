#include <cuda_runtime.h>
#include <stdlib.h>
struct Image {
  int width;
  int height;
  unsigned char * r;
  unsigned char * g;
  unsigned char * b;
  unsigned char onDevice;
};

void allocationImage(struct Image * image, unsigned char onDevice) {
  unsigned int pixelCount = (*image).width * (*image).height;
  if (onDevice) {
    cudaMalloc((void**)&(*image).r, pixelCount * sizeof(unsigned char));
    cudaMalloc((void**)&(*image).g, pixelCount * sizeof(unsigned char));
    cudaMalloc((void**)&(*image).b, pixelCount * sizeof(unsigned char));
  } else {
    (*image).r = malloc(pixelCount * sizeof(unsigned char));
    (*image).g = malloc(pixelCount * sizeof(unsigned char));
    (*image).b = malloc(pixelCount * sizeof(unsigned char));
  }
}
