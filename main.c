#include "cuda.cuh"
#include "qdbmp.h"
#include <cuda_device_runtime_api.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>

int main() {
  printf("Program start.\n");
  test();
  BMP_GetError();
  const char* inFile = "okanagan.bmp";
	const char* outFile = "okanagan_processed2.bmp";

	UINT width, height;
	UINT x, y;
	BMP* bmp;

	/* Read an image file */
	bmp = BMP_ReadFile(inFile);
	BMP_CHECK_ERROR(stdout, -1);

  printf("Image loaded.\n");

	/* Get image's dimensions */
	width = BMP_GetWidth(bmp);
	height = BMP_GetHeight(bmp);
  unsigned int pixelCount = width * height;
  BMP* bmpOut = BMP_Create(width, height, 32);
  BMP_CHECK_ERROR( stderr, -2 );

  printf("Width: %d, Height: %d\n", (int)width, (int)height);

  UCHAR * h_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR * h_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR * h_b = malloc(pixelCount * sizeof(UCHAR));

  UCHAR * d_r;
  UCHAR * d_g;
  UCHAR * d_b;
  cudaMalloc((void**)&d_r, pixelCount * sizeof(UCHAR));
  cudaMalloc((void**)&d_g, pixelCount * sizeof(UCHAR));
  cudaMalloc((void**)&d_b, pixelCount * sizeof(UCHAR));
  
  printf("H/D memory allocation done.\n");

	/* Iterate through all the image's pixels */
	for (y = 0; y < height; ++y) {
		for (x = 0; x < width; ++x) {
			/* Get pixel's RGB values */
      int index = x + y * width;
			BMP_GetPixelRGB(bmp, x, y, h_r+index, h_g+index, h_b+index);
		}
	}

  cudaMemcpy(d_r, h_r, pixelCount * sizeof(UCHAR), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g, h_g, pixelCount * sizeof(UCHAR), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, pixelCount * sizeof(UCHAR), cudaMemcpyHostToDevice);
  printf("Data sent to device.\n");

	for (y = 0; y < height; ++y) {
		for (x = 0; x < width; ++x) {
			/* Get pixel's RGB values */
      int index = x + y * width;
      *(h_r+index) = *(h_r+index)/10;
      *(h_g+index) = *(h_g+index)/10;
      *(h_b+index) = *(h_b+index)/10;
		}
	}

  negative(d_r, d_g, d_b, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(h_r, d_r, pixelCount * sizeof(UCHAR), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_g, d_g, pixelCount * sizeof(UCHAR), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, pixelCount * sizeof(UCHAR), cudaMemcpyDeviceToHost);
  printf("Data received from device.\n");

	for (y = 0; y < height; ++y) {
		for (x = 0; x < width; ++x) {
			/* Get pixel's RGB values */
      int index = x + y * width;
      BMP_SetPixelRGB(bmpOut, x, y, *(h_r+index), *(h_g+index), *(h_b+index));
		}
	}

	/* Save result */
	BMP_WriteFile(bmpOut, outFile);
	BMP_CHECK_ERROR(stdout, -3);

	/* Free all memory allocated for the image */
	BMP_Free(bmp);
  cudaFree(d_r);
  cudaFree(d_g);
  cudaFree(d_b);
  free(h_r);
  free(h_g);
  free(h_b);

	return 0;
}
