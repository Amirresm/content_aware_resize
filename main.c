#include "cuda.cuh"
#include "qdbmp.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
  int x;
  int y;
} point;

int dfs_vertical(int x, int y, UCHAR *gray_channel, int bound_low,
                 int bound_high, int width, int height, int total_energy,
                 point *path) {
  int index = x + y * width;
  // gray_channel[index] = 0;
  total_energy += gray_channel[index];
  if (y == 0) {
    path[y].x = x;
    path[y].y = y;
  }
  if (y == height - 1) {
    return total_energy;
  }
  int children_energy[] = {0, 0, 0};
  int least_energy = 99999999;
  int least_child_x = 0;
  int i = 0;
  for (int x_child = x - 1; x_child <= x + 1; ++x_child) {
    if (x_child < bound_low || x_child >= bound_high) {
      continue;
    }
    int next_energy =
        dfs_vertical(x_child, y + 1, gray_channel, bound_low, bound_high, width,
                     height, total_energy, path);
    children_energy[i] = next_energy;
    i++;
  }
  for (int i = 0; i < 3; i++) {
    if (children_energy[i] < least_energy) {
      least_energy = children_energy[i];
      least_child_x = x + i - 1;
    }
  }
  path[y].x = least_child_x;
  path[y].y = y + 1;
  return least_energy;
}

void find_best_path1(int width, int height, UCHAR *h_r,
                     point *least_energy_paths) {
  int least_total_energy = 99999999;
  for (int batch = 0; batch < 1; batch++) {
    int x = rand() % width;
    point *path = malloc(height * sizeof(point));
    int total_energy =
        dfs_vertical(x, 0, h_r, x - 1, x + 1, width, height, 0, path);
    if (total_energy < least_total_energy) {
      least_total_energy = total_energy;
      memcpy(least_energy_paths, path, height * sizeof(point));
    }
  }
}
void find_best_path2(int width, int height, int batch_size, UCHAR *h_r,
                     point *least_energy_paths) {
  int least_total_energy = 99999999;
  for (int batch = 0; batch < batch_size; batch++) {
    // int x = (width / 100) * batch;
    int x = rand() % width;
    int total_energy = 0;
    point *path = malloc(height * sizeof(point));

    for (int y = 0; y < height; ++y) {
      int index = x + y * width;
      UCHAR energy = h_r[index];
      total_energy += energy * energy;
      path[y].x = x;
      path[y].y = y;
      int min_child_energy = 255;
      int rng = rand() % 31231412;

      for (int j = 0; j < 3; ++j) {
        int x_child = x + (rng + j) % 3 - 1;
        if (x_child < 0 || x_child >= width) {
          continue;
        }
        int index_child = x_child + (y + 1) * width;
        UCHAR energy_child = h_r[index_child];
        if (energy_child < min_child_energy) {
          min_child_energy = energy_child;
          x = x_child;
        }
      }
    }
    if (total_energy < least_total_energy) {
      least_total_energy = total_energy;
      memcpy(least_energy_paths, path, height * sizeof(point));
    }
  }
}

int main() {
  printf("Program start.\n");
  srand(time(NULL));

  BMP_GetError();
  const char *inFileBase = "main";
  char inFile[100];
  sprintf(inFile, "%s.bmp", inFileBase);
  char outColFile[100];
  char outEnergyFile[100];
  char outCroppedFile[100];
  sprintf(outColFile, "%s_out_col.bmp", inFileBase);
  sprintf(outEnergyFile, "%s_out_energy.bmp", inFileBase);
  sprintf(outCroppedFile, "%s_out_cropped.bmp", inFileBase);

  int crop_percent = 25;
  int batch_size = 10;

  UINT width, height;
  UINT x, y;
  BMP *bmp;

  bmp = BMP_ReadFile(inFile);
  BMP_CHECK_ERROR(stdout, -1);

  printf("Image loaded.\n");

  width = BMP_GetWidth(bmp);
  height = BMP_GetHeight(bmp);
  unsigned int pixelCount = width * height;
  BMP_CHECK_ERROR(stderr, -2);

  printf("Width: %d, Height: %d\n", (int)width, (int)height);

  UCHAR *original_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *original_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *original_b = malloc(pixelCount * sizeof(UCHAR));

  UCHAR *energy_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *energy_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *energy_b = malloc(pixelCount * sizeof(UCHAR));

  UCHAR *cropped_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_b = malloc(pixelCount * sizeof(UCHAR));

  printf("H/D memory allocation done.\n");

  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      int index = x + y * width;
      BMP_GetPixelRGB(bmp, x, y, original_r + index, original_g + index,
                      original_b + index);
      BMP_GetPixelRGB(bmp, x, y, energy_r + index, energy_g + index,
                      energy_b + index);
      BMP_GetPixelRGB(bmp, x, y, cropped_r + index, cropped_g + index,
                      cropped_b + index);
    }
  }

  printf("Data sent to device.\n");

  energy(energy_r, energy_g, energy_b, width, height);

  int desaturation = 5;
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      int index = x + y * width;
      *(energy_r + index) = *(energy_r + index) / desaturation;
      *(energy_g + index) = *(energy_g + index) / desaturation;
      *(energy_b + index) = *(energy_b + index) / desaturation;
    }
  }

  printf("Data received from device.\n");

  int n_cols = width * crop_percent / 100;
  point **least_energy_paths = malloc(n_cols * sizeof(point *));
  for (int i = 0; i < n_cols; i++) {
    least_energy_paths[i] = malloc(height * sizeof(point));
    find_best_path2(width, height, batch_size, energy_r, least_energy_paths[i]);
    for (int y = 0; y < height; ++y) {
      int x = least_energy_paths[i][y].x;
      int index = x + y * width;
      energy_r[index] = 255;
      energy_g[index] = 0;
      energy_b[index] = 0;
    }
  }
  for (int i = 0; i < n_cols; i++) {
    for (int y = 0; y < height; ++y) {
      int x = least_energy_paths[i][y].x;
      int index = x + y * width;
      original_r[index] = 255;
      original_g[index] = 0;
      original_b[index] = 0;
    }
    free(least_energy_paths[i]);
  }

  BMP *bmpEnergy = BMP_Create(width, height, 32);
  BMP *bmpOut = BMP_Create(width, height, 32);
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      int index = x + y * width;
      BMP_SetPixelRGB(bmpEnergy, x, y, *(energy_r + index), *(energy_g + index),
                      *(energy_b + index));
      BMP_SetPixelRGB(bmpOut, x, y, *(original_r + index),
                      *(original_g + index), *(original_b + index));
    }
  }
  BMP *bmpCropped = BMP_Create(width - n_cols, height, 32);
  for (y = 0; y < height; ++y) {
    int reducer = 0;
    for (x = 0; x < width; ++x) {
      int index = x + y * width;
      if (energy_r[index] == 255 && energy_g[index] == 0 &&
          energy_b[index] == 0) {
        reducer++;
        continue;
      }
      BMP_SetPixelRGB(bmpCropped, x - reducer, y, *(original_r + index),
                      *(original_g + index), *(original_b + index));
    }
  }

  BMP_WriteFile(bmpOut, outColFile);
  BMP_CHECK_ERROR(stdout, -3);
  BMP_WriteFile(bmpEnergy, outEnergyFile);
  BMP_CHECK_ERROR(stdout, -3);
  BMP_WriteFile(bmpCropped, outCroppedFile);
  BMP_CHECK_ERROR(stdout, -3);

  BMP_Free(bmp);
  cudaFree(energy_r);
  cudaFree(energy_g);
  cudaFree(energy_b);
  free(energy_r);
  free(energy_g);
  free(energy_b);

  return 0;
}
