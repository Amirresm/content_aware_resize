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

void copy_rgb(UCHAR *h_r, UCHAR *h_g, UCHAR *h_b, UCHAR *d_r, UCHAR *d_g,
              UCHAR *d_b, int width, int height) {
  int pixelCount = width * height;
  memcpy(h_r, d_r, pixelCount * sizeof(UCHAR));
  memcpy(h_g, d_g, pixelCount * sizeof(UCHAR));
  memcpy(h_b, d_b, pixelCount * sizeof(UCHAR));
}

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
int find_x_of_unremoved(int x, int y, int dir, UCHAR *removed_mask, int width,
                        int height) {
  int index = x + y * width;
  if (removed_mask[index] == 0 || x <= 0 || x >= width - 1) {
    return x;
  }
  if (dir == 0) {
    return find_x_of_unremoved(x - 1, y, dir, removed_mask, width, height);
  } else {
    return find_x_of_unremoved(x + 1, y, dir, removed_mask, width, height);
  }
}
int find_y_of_unremoved(int x, int y, int dir, UCHAR *removed_mask, int width,
                        int height) {
  int index = x + y * width;
  if (removed_mask[index] == 0 || y <= 0 || y >= height - 1) {
    return y;
  }
  if (dir == 0) {
    return find_y_of_unremoved(x, y - 1, dir, removed_mask, width, height);
  } else {
    return find_y_of_unremoved(x, y + 1, dir, removed_mask, width, height);
  }
}
void find_best_vertical_seam(int width, int height, int batch_size,
                             UCHAR *energy_matrix, point *least_energy_paths,
                             UCHAR *removed_mask) {
  int least_total_energy_disruption = 99999999;
  for (int batch = 0; batch < batch_size; batch++) {
    int x = rand() % width;
    int total_energy_disruption = 0;
    point *path = malloc(height * sizeof(point));

    for (int y = 0; y < height; ++y) {
      int index = x + y * width;
      path[y].x = x;
      path[y].y = y;
      int min_energy_disrupt = 9999999;
      int rng = rand() % 31231412;

      int left_x =
          find_x_of_unremoved(x - 1, y + 1, 0, removed_mask, width, height);
      int center_x =
          find_x_of_unremoved(x, y + 1, 1, removed_mask, width, height);
      int right_x = find_x_of_unremoved(center_x + 1, y + 1, 1, removed_mask,
                                        width, height);
      for (int j = 0; j < 3; ++j) {
        char random_child_index = (rng + j) % 3 - 1; // random order of -1, 0, 1
        int x_child;
        int introduced_energy = 0;
        switch (random_child_index) {
        case -1: // left
          x_child = left_x;
          if (x_child <= 1 || y == height - 1) {
            introduced_energy = 255;
          } else {
            int x_left_parent =
                find_x_of_unremoved(x - 1, y, 0, removed_mask, width, height);
            int x_right_child = find_x_of_unremoved(
                x_child + 1, y + 1, 1, removed_mask, width, height);
            ;
            int x_left_child = find_x_of_unremoved(x_child - 1, y + 1, 0,
                                                   removed_mask, width, height);
            ;
            introduced_energy =
                abs(energy_matrix[x_right_child + (y + 1) * width] -
                    energy_matrix[x_left_parent + y * width]) +
                abs(energy_matrix[x_right_child + (y + 1) * width] -
                    energy_matrix[x_left_child + (y + 1) * width]);
          }
          break;

        case 0: // center
          x_child = center_x;
          if (x_child == 0 || x_child == width - 1 || y == height - 1) {
            introduced_energy = 255;
          } else {
            int x_right_child = find_x_of_unremoved(
                x_child + 1, y + 1, 1, removed_mask, width, height);
            ;
            int x_left_child = find_x_of_unremoved(x_child - 1, y + 1, 0,
                                                   removed_mask, width, height);
            introduced_energy =
                abs(energy_matrix[x_right_child + (y + 1) * width] -
                    energy_matrix[x_left_child + (y + 1) * width]);
          }
          break;

        case 1: // right
          x_child = right_x;
          if (x_child >= width - 2 || y == height - 1) {
            introduced_energy = 255;
          } else {
            int x_right_parent =
                find_x_of_unremoved(x + 1, y, 1, removed_mask, width, height);
            int x_right_child = find_x_of_unremoved(
                x_child + 1, y + 1, 1, removed_mask, width, height);
            ;
            int x_left_child = find_x_of_unremoved(x_child - 1, y + 1, 0,
                                                   removed_mask, width, height);
            ;
            introduced_energy =
                abs(energy_matrix[x_left_child + (y + 1) * width] -
                    energy_matrix[x_right_parent + y * width]) +
                abs(energy_matrix[x_left_child + (y + 1) * width] -
                    energy_matrix[x_right_child + (y + 1) * width]);
          }
          break;
        }
        if (introduced_energy < min_energy_disrupt) {
          min_energy_disrupt = introduced_energy;
          x = x_child;
        }
      }

      total_energy_disruption += min_energy_disrupt;
    }
    if (total_energy_disruption < least_total_energy_disruption) {
      least_total_energy_disruption = total_energy_disruption;
      memcpy(least_energy_paths, path, height * sizeof(point));
    }
  }
}
void find_best_horiz_seam(int width, int height, int batch_size,
                          UCHAR *energy_matrix, point *least_energy_paths,
                          UCHAR *removed_mask) {
  int least_total_energy_disruption = 99999999;
  for (int batch = 0; batch < batch_size; batch++) {
    int y = rand() % height;
    int total_energy_disruption = 0;
    point *path = malloc(width * sizeof(point));

    for (int x = 0; x < width; ++x) {
      int index = x + y * width;
      path[x].x = x;
      path[x].y = y;
      int min_energy_disrupt = 9999999;
      int rng = rand() % 31231412;

      int up_y =
          find_y_of_unremoved(x + 1, y - 1, 0, removed_mask, width, height);
      int center_y =
          find_y_of_unremoved(x + 1, y, 1, removed_mask, width, height);
      int down_y = find_y_of_unremoved(x + 1, center_y + 1, 1, removed_mask,
                                       width, height);
      for (int j = 0; j < 3; ++j) {
        char random_child_index = (rng + j) % 3 - 1; // random order of -1, 0, 1
        int y_child;
        int introduced_energy = 0;
        switch (random_child_index) {
        case -1: // up
          y_child = up_y;
          if (y_child <= 1 || x == width - 1) {
            introduced_energy = 255;
          } else {
            int y_up_parent =
                find_y_of_unremoved(x, y - 1, 0, removed_mask, width, height);
            int y_down_child = find_y_of_unremoved(x + 1, y_child + 1, 1,
                                                   removed_mask, width, height);
            ;
            int y_up_child = find_y_of_unremoved(x + 1, y_child - 1, 0,
                                                 removed_mask, width, height);
            ;
            introduced_energy =
                abs(energy_matrix[x + 1 + y_up_child * width] -
                    energy_matrix[x + y_up_parent * width]) +
                abs(energy_matrix[x + 1 + (y_down_child)*width] -
                    energy_matrix[x + 1 + (y_up_child)*width]);
            // introduced_energy = energy_matrix[x + 1 + y_child * width];
          }
          break;

        case 0: // center
          y_child = center_y;
          if (y_child == 0 || y_child == height - 1 || x == width - 1) {
            introduced_energy = 255;
          } else {
            int y_down_child = find_y_of_unremoved(x + 1, y_child + 1, 1,
                                                   removed_mask, width, height);
            ;
            int y_up_child = find_y_of_unremoved(x + 1, y_child - 1, 0,
                                                 removed_mask, width, height);
            introduced_energy =
                abs(energy_matrix[x + 1 + (y_down_child)*width] -
                    energy_matrix[x + 1 + (y_up_child)*width]);
            // introduced_energy = energy_matrix[x + 1 + y_child * width];
          }
          break;

        case 1: // right
          y_child = down_y;
          if (y_child >= height - 2 || x == width - 1) {
            introduced_energy = 255;
          } else {
            int y_down_parent = find_y_of_unremoved(
                x, y_child + 1, 1, removed_mask, width, height);
            int y_down_child = find_y_of_unremoved(x + 1, y_child + 1, 1,
                                                   removed_mask, width, height);
            ;
            int y_up_child = find_y_of_unremoved(x + 1, y_child - 1, 0,
                                                 removed_mask, width, height);
            ;
            introduced_energy =
                abs(energy_matrix[x + 1 + y_down_child * width] -
                    energy_matrix[x + y_down_parent * width]) +
                abs(energy_matrix[x + 1 + y_up_child * width] -
                    energy_matrix[x + 1 + y_down_child * width]);
            // introduced_energy = energy_matrix[x + 1 + y_child * width];
          }
          break;
        }
        if (introduced_energy < min_energy_disrupt) {
          min_energy_disrupt = introduced_energy;
          y = y_child;
        }
      }

      total_energy_disruption += min_energy_disrupt;
    }
    if (total_energy_disruption < least_total_energy_disruption) {
      least_total_energy_disruption = total_energy_disruption;
      memcpy(least_energy_paths, path, width * sizeof(point));
    }
  }
}

void update_energy(int x, int y, UCHAR *energy_matrix, UCHAR *removed_mask,
                   UCHAR *r, UCHAR *g, UCHAR *b, int width, int height) {
  int x_left = find_x_of_unremoved(x - 1, y, 0, removed_mask, width, height);
  int x_right = find_x_of_unremoved(x + 1, y, 1, removed_mask, width, height);
  int y_up = find_y_of_unremoved(x, y - 1, 0, removed_mask, width, height);
  int y_down = find_y_of_unremoved(x, y + 1, 1, removed_mask, width, height);

  UCHAR e_r = (abs(r[x_left + y * width] - r[x + y * width]) +
               abs(r[x_right + y * width] - r[x + y * width]) +
               abs(r[x + y_up * width] - r[x + y * width]) +
               abs(r[x + y_down * width] - r[x + y * width])) /
              4;
  UCHAR e_g = (abs(g[x_left + y * width] - g[x + y * width]) +
               abs(g[x_right + y * width] - g[x + y * width]) +
               abs(g[x + y_up * width] - g[x + y * width]) +
               abs(g[x + y_down * width] - g[x + y * width])) /
              4;
  UCHAR e_b = (abs(b[x_left + y * width] - b[x + y * width]) +
               abs(b[x_right + y * width] - b[x + y * width]) +
               abs(b[x + y_up * width] - b[x + y * width]) +
               abs(b[x + y_down * width] - b[x + y * width])) /
              4;
  energy_matrix[x + y * width] = e_r * 0.3 + e_g * 0.59 + e_b * 0.11;
}

int main() {
  printf("Program start.\n");
  srand(time(NULL));

  BMP_GetError();
  const char *inFileBase = "main";
  // const char *inFileBase = "okanagan";
  char inFile[100];
  sprintf(inFile, "%s.bmp", inFileBase);
  char outColFile[100];
  char outEnergyFile[100];
  char outCroppedFile[100];
  sprintf(outColFile, "%s_out_col.bmp", inFileBase);
  sprintf(outEnergyFile, "%s_out_energy.bmp", inFileBase);
  sprintf(outCroppedFile, "%s_out_cropped.bmp", inFileBase);

  int crop_percent = 25;
  int batch_size = 100;

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

  UCHAR *removed_mask = malloc(pixelCount * sizeof(UCHAR));
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      int index = x + y * width;
      BMP_GetPixelRGB(bmp, x, y, original_r + index, original_g + index,
                      original_b + index);

      removed_mask[index] = 0;

      // rescaling to 254 to use 255 as a removed seam marker
      // original_r[index] = (original_r[index] / 255) * 254;
    }
  }

  UCHAR *energy_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *energy_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *energy_b = malloc(pixelCount * sizeof(UCHAR));

  UCHAR *cropped_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_b = malloc(pixelCount * sizeof(UCHAR));

  copy_rgb(energy_r, energy_g, energy_b, original_r, original_g, original_b,
           width, height);
  copy_rgb(cropped_r, cropped_g, cropped_b, original_r, original_g, original_b,
           width, height);

  printf("H/D memory allocation done.\n");

  printf("Data sent to device.\n");

  energy(energy_r, energy_g, energy_b, width, height);

  printf("Data received from device.\n");

  // int n_cols = width * crop_percent / 100;
  // point **least_energy_paths = malloc(n_cols * sizeof(point *));
  // for (int i = 0; i < n_cols; i++) {
  //   least_energy_paths[i] = malloc(height * sizeof(point));
  //   find_best_vertical_seam(width, height, batch_size, energy_r,
  //                           least_energy_paths[i], removed_mask);
  //   for (int y = 0; y < height; ++y) {
  //     int x = least_energy_paths[i][y].x;
  //     int index = x + y * width;

  //     removed_mask[index] = 255;

  //     int x_left =
  //         find_x_of_unremoved(x - 1, y, 0, removed_mask, width, height);
  //     int x_right =
  //         find_x_of_unremoved(x + 1, y, 1, removed_mask, width, height);
  //     int y_up = find_y_of_unremoved(x, y - 1, 0, removed_mask, width,
  //     height); int y_down =
  //         find_y_of_unremoved(x, y + 1, 1, removed_mask, width, height);
  //     update_energy(x_left, y, energy_r, removed_mask, original_r,
  //     original_g,
  //                   original_b, width, height);
  //     update_energy(x_right, y, energy_r, removed_mask, original_r,
  //     original_g,
  //                   original_b, width, height);
  //     update_energy(x, y_up, energy_r, removed_mask, original_r, original_g,
  //                   original_b, width, height);
  //     update_energy(x, y_down, energy_r, removed_mask, original_r,
  //     original_g,
  //                   original_b, width, height);
  //   }
  // }
  // for (int i = 0; i < n_cols; i++) {
  //   for (int y = 0; y < height; ++y) {
  //     int x = least_energy_paths[i][y].x;
  //     int index = x + y * width;
  //     original_r[index] = 255;
  //     original_g[index] = 0;
  //     original_b[index] = 0;
  //   }
  // }

  int n_rows = height * crop_percent / 100;
  point **least_energy_paths = malloc(n_rows * sizeof(point *));
  for (int i = 0; i < n_rows; i++) {
    least_energy_paths[i] = malloc(width * sizeof(point));
    find_best_horiz_seam(width, height, batch_size, energy_r,
                         least_energy_paths[i], removed_mask);
    for (int x = 0; x < width; ++x) {
      int y = least_energy_paths[i][x].y;
      int index = x + y * width;

      removed_mask[index] = 255;

      int x_left =
          find_x_of_unremoved(x - 1, y, 0, removed_mask, width, height);
      int x_right =
          find_x_of_unremoved(x + 1, y, 1, removed_mask, width, height);
      int y_up = find_y_of_unremoved(x, y - 1, 0, removed_mask, width, height);
      int y_down =
          find_y_of_unremoved(x, y + 1, 1, removed_mask, width, height);
      update_energy(x_left, y, energy_r, removed_mask, original_r, original_g,
                    original_b, width, height);
      update_energy(x_right, y, energy_r, removed_mask, original_r, original_g,
                    original_b, width, height);
      update_energy(x, y_up, energy_r, removed_mask, original_r, original_g,
                    original_b, width, height);
      update_energy(x, y_down, energy_r, removed_mask, original_r, original_g,
                    original_b, width, height);
    }
  }
  for (int i = 0; i < n_rows; i++) {
    for (int x = 0; x < width; ++x) {
      int y = least_energy_paths[i][x].y;
      int index = x + y * width;
      original_r[index] = 255;
      original_g[index] = 0;
      original_b[index] = 0;
    }
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
  // BMP *bmpCropped = BMP_Create(width - n_rows, height, 32);
  // for (y = 0; y < height; ++y) {
  //   int reducer = 0;
  //   for (x = 0; x < width; ++x) {
  //     int index = x + y * width;
  //     if (removed_mask[index] == 255) {
  //       reducer++;
  //       continue;
  //     }
  //     BMP_SetPixelRGB(bmpCropped, x - reducer, y, *(original_r + index),
  //                     *(original_g + index), *(original_b + index));
  //   }
  // }
  BMP *bmpCropped = BMP_Create(width, height - n_rows, 32);
  for (x = 0; x < width; ++x) {
    int reducer = 0;
    for (y = 0; y < height; ++y) {
      int index = x + y * width;
      if (removed_mask[index] == 255) {
        reducer++;
        continue;
      }
      BMP_SetPixelRGB(bmpCropped, x, y - reducer, *(original_r + index),
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
  // free(energy_r);
  // free(energy_g);
  // free(energy_b);

  return 0;
}
