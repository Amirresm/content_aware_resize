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

void find_horiz_of_unremoved(point *p, int dir, UCHAR *removed_mask, int width,
                             int height) {
  int x = p->x;
  int y = p->y;
  int index = x + y * width;
  if (removed_mask[index] == 0 || x <= 0 || x >= width - 1) {
    return;
  }
  if (removed_mask[index] == 1) {
    if (dir == 0) {
      p->x -= 1;
      return find_horiz_of_unremoved(p, dir, removed_mask, width, height);
    } else {
      p->x += 1;
      return find_horiz_of_unremoved(p, dir, removed_mask, width, height);
    }
  } else {
    p->y += 1;
    return find_horiz_of_unremoved(p, dir, removed_mask, width, height);
  }
}
void find_vert_of_unremoved(point *p, int dir, UCHAR *removed_mask, int width,
                            int height) {
  int x = p->x;
  int y = p->y;
  int index = x + y * width;
  if (removed_mask[index] == 0 || y <= 0 || y >= height - 1) {
    return;
  }
  if (removed_mask[index] == 2) {
    if (dir == 0) {
      p->y -= 1;
      return find_vert_of_unremoved(p, dir, removed_mask, width, height);
    } else {
      p->y += 1;
      return find_vert_of_unremoved(p, dir, removed_mask, width, height);
    }
  } else {
    p->x += 1;
    return find_vert_of_unremoved(p, dir, removed_mask, width, height);
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

      point left_c = {.x = x - 1, .y = y + 1};
      find_horiz_of_unremoved(&left_c, 0, removed_mask, width, height);
      point center_c = {.x = x, .y = y + 1};
      find_horiz_of_unremoved(&center_c, 1, removed_mask, width, height);
      point right_c = {.x = center_c.x + 1, .y = y + 1};
      find_horiz_of_unremoved(&right_c, 1, removed_mask, width, height);

      for (int j = 0; j < 3; ++j) {
        char random_child_index = (rng + j) % 3 - 1; // random order of -1, 0, 1
        // int x_child;
        point child;
        int introduced_energy = 0;
        switch (random_child_index) {
        case -1: // left
          // x_child = left_x;
          child.x = left_c.x;
          child.y = left_c.y;
          if (child.x <= 1 || y == height - 1) {
            introduced_energy = 255;
          } else {
            point left_parent = {.x = x - 1, .y = y};
            find_horiz_of_unremoved(&left_parent, 0, removed_mask, width,
                                    height);
            point right_child = {.x = child.x + 1, .y = y + 1};
            find_horiz_of_unremoved(&right_child, 1, removed_mask, width,
                                    height);
            point left_child = {.x = child.x - 1, .y = y + 1};
            find_horiz_of_unremoved(&left_child, 0, removed_mask, width,
                                    height);

            introduced_energy =
                abs(energy_matrix[right_child.x + right_child.y * width] -
                    energy_matrix[left_parent.x + left_parent.y * width]) +
                abs(energy_matrix[right_child.x + right_child.y * width] -
                    energy_matrix[left_child.x + left_child.y * width]);
          }
          break;

        case 0: // center
          // x_child = center_x;
          child.x = center_c.x;
          child.y = center_c.y;
          if (child.x == 0 || child.x == width - 1 || y == height - 1) {
            introduced_energy = 255;
          } else {
            point right_child = {.x = child.x + 1, .y = y + 1};
            find_horiz_of_unremoved(&right_child, 1, removed_mask, width,
                                    height);
            point left_child = {.x = child.x - 1, .y = y + 1};
            find_horiz_of_unremoved(&left_child, 0, removed_mask, width,
                                    height);
            introduced_energy =
                abs(energy_matrix[right_child.x + right_child.y * width] -
                    energy_matrix[left_child.x + left_child.y * width]);
          }
          break;

        case 1: // right
          // x_child = right_x;
          child.x = right_c.x;
          child.y = right_c.y;
          if (child.x >= width - 2 || y == height - 1) {
            introduced_energy = 255;
          } else {
            point right_parent = {.x = x + 1, .y = y};
            find_horiz_of_unremoved(&right_parent, 0, removed_mask, width,
                                    height);
            point right_child = {.x = child.x + 1, .y = y + 1};
            find_horiz_of_unremoved(&right_child, 1, removed_mask, width,
                                    height);
            point left_child = {.x = child.x - 1, .y = y + 1};
            find_horiz_of_unremoved(&left_child, 0, removed_mask, width,
                                    height);
            introduced_energy =
                abs(energy_matrix[left_child.x + left_child.y * width] -
                    energy_matrix[right_parent.x + right_parent.y * width]) +
                abs(energy_matrix[left_child.x + left_child.y * width] -
                    energy_matrix[right_child.x + right_child.y * width]);
          }
          break;
        }
        if (introduced_energy < min_energy_disrupt) {
          min_energy_disrupt = introduced_energy;
          x = child.x;
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

      point up_c = {.x = x + 1, .y = y - 1};
      find_vert_of_unremoved(&up_c, 0, removed_mask, width, height);

      point center_c = {.x = x + 1, .y = y};
      find_vert_of_unremoved(&center_c, 1, removed_mask, width, height);

      point down_c = {.x = x + 1, .y = center_c.y + 1};
      find_vert_of_unremoved(&down_c, 1, removed_mask, width, height);

      for (int j = 0; j < 3; ++j) {
        char random_child_index = (rng + j) % 3 - 1; // random order of -1, 0, 1
        // int y_child;
        point child;
        int introduced_energy = 0;
        switch (random_child_index) {
        case -1: // up
          // y_child = up_y;
          child.x = up_c.x;
          child.y = up_c.y;
          if (child.y <= 1 || x == width - 1) {
            introduced_energy = 255;
          } else {
            point up_parent = {.x = x, .y = y - 1};
            find_vert_of_unremoved(&up_parent, 0, removed_mask, width, height);
            point down_child = {.x = x + 1, .y = child.y + 1};
            find_vert_of_unremoved(&down_child, 1, removed_mask, width, height);

            point up_child = {.x = x + 1, .y = child.y - 1};
            find_vert_of_unremoved(&up_child, 0, removed_mask, width, height);

            introduced_energy =
                abs(energy_matrix[up_child.x + up_child.y * width] -
                    energy_matrix[up_parent.x + up_parent.y * width]) +
                abs(energy_matrix[down_child.x + down_child.y * width] -
                    energy_matrix[up_child.x + up_child.y * width]);
            // introduced_energy = energy_matrix[x + 1 + y_child * width];
          }
          break;

        case 0: // center
          // y_child = center_y;
          child.x = center_c.x;
          child.y = center_c.y;

          if (child.y == 0 || child.y == height - 1 || x == width - 1) {
            introduced_energy = 255;
          } else {
            point down_child = {.x = x + 1, .y = child.y + 1};
            find_vert_of_unremoved(&down_child, 1, removed_mask, width, height);
            point up_child = {.x = x + 1, .y = child.y - 1};
            find_vert_of_unremoved(&up_child, 0, removed_mask, width, height);
            introduced_energy =
                abs(energy_matrix[down_child.x + down_child.y * width] -
                    energy_matrix[up_child.x + up_child.y * width]);
            // introduced_energy = energy_matrix[x + 1 + y_child * width];
          }
          break;

        case 1: // right
          // y_child = down_y;
          child.x = down_c.x;
          child.y = down_c.y;
          if (child.y >= height - 2 || x == width - 1) {
            introduced_energy = 255;
          } else {
            point down_parent = {.x = x, .y = y + 1};
            find_vert_of_unremoved(&down_parent, 1, removed_mask, width,
                                   height);
            point down_child = {.x = x + 1, .y = child.y + 1};
            find_vert_of_unremoved(&down_child, 1, removed_mask, width, height);
            point up_child = {.x = x + 1, .y = child.y - 1};
            find_vert_of_unremoved(&up_child, 0, removed_mask, width, height);
            introduced_energy =
                abs(energy_matrix[down_child.x + down_child.y * width] -
                    energy_matrix[down_parent.x + down_parent.y * width]) +
                abs(energy_matrix[up_child.x + up_child.y * width] -
                    energy_matrix[down_child.x + down_child.y * width]);
            // introduced_energy = energy_matrix[x + 1 + y_child * width];
          }
          break;
        }
        if (introduced_energy < min_energy_disrupt) {
          min_energy_disrupt = introduced_energy;
          y = child.y;
        }
        // if (removed_mask[x + y * width] != 0) {
        //   printf("ERROR: removed mask is 1\n");
        // }
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
  point left = {.x = x - 1, .y = y};
  find_horiz_of_unremoved(&left, 0, removed_mask, width, height);
  point right = {.x = x + 1, .y = y};
  find_horiz_of_unremoved(&right, 1, removed_mask, width, height);
  point up = {.x = x, .y = y - 1};
  find_vert_of_unremoved(&up, 0, removed_mask, width, height);
  point down = {.x = x, .y = y + 1};
  find_vert_of_unremoved(&down, 1, removed_mask, width, height);

  UCHAR e_r = (abs(r[left.x + left.y * width] - r[x + y * width]) +
               abs(r[right.x + right.y * width] - r[x + y * width]) +
               abs(r[up.x + up.y * width] - r[x + y * width]) +
               abs(r[down.x + down.y * width] - r[x + y * width])) /
              4;
  UCHAR e_g = (abs(g[left.x + left.y * width] - g[x + y * width]) +
               abs(g[right.x + right.y * width] - g[x + y * width]) +
               abs(g[up.x + up.y * width] - g[x + y * width]) +
               abs(g[down.x + down.y * width] - g[x + y * width])) /
              4;
  UCHAR e_b = (abs(b[left.x + left.y * width] - b[x + y * width]) +
               abs(b[right.x + right.y * width] - b[x + y * width]) +
               abs(b[up.x + up.y * width] - b[x + y * width]) +
               abs(b[down.x + down.y * width] - b[x + y * width])) /
              4;
  energy_matrix[x + y * width] = e_r * 0.3 + e_g * 0.59 + e_b * 0.11;
}

void update_enery_around_pixel(int x, int y, UCHAR *energy_matrix,
                               UCHAR *removed_mask, UCHAR *r, UCHAR *g,
                               UCHAR *b, int width, int height) {
  point left = {.x = x - 1, .y = y};
  find_horiz_of_unremoved(&left, 0, removed_mask, width, height);
  point right = {.x = x + 1, .y = y};
  find_horiz_of_unremoved(&right, 1, removed_mask, width, height);
  point up = {.x = x, .y = y - 1};
  find_vert_of_unremoved(&up, 0, removed_mask, width, height);
  point down = {.x = x, .y = y + 1};
  find_vert_of_unremoved(&down, 1, removed_mask, width, height);
  update_energy(left.x, left.y, energy_matrix, removed_mask, r, g, b, width,
                height);
  update_energy(right.x, right.y, energy_matrix, removed_mask, r, g, b, width,
                height);
  update_energy(up.x, up.y, energy_matrix, removed_mask, r, g, b, width,
                height);
  update_energy(up.x, down.y, energy_matrix, removed_mask, r, g, b, width,
                height);
}

void remove_vert_seam(int width, int height, int batch_size, UCHAR *energy_r,
                      UCHAR *removed_mask, UCHAR *original_r, UCHAR *original_g,
                      UCHAR *original_b) {
  point *vert_least_energy_paths = malloc(height * sizeof(point));
  find_best_vertical_seam(width, height, batch_size, energy_r,
                          vert_least_energy_paths, removed_mask);
  for (int y = 0; y < height; ++y) {
    int x = vert_least_energy_paths[y].x;
    int index = x + y * width;

    if (removed_mask[index] == 0)
      removed_mask[index] = 1;
    else
      removed_mask[index] = 3;

    update_enery_around_pixel(x, y, energy_r, removed_mask, original_r,
                              original_g, original_b, width, height);
  }
  free(vert_least_energy_paths);
}

void remove_horiz_seam(int width, int height, int batch_size, UCHAR *energy_r,
                       UCHAR *removed_mask, UCHAR *original_r,
                       UCHAR *original_g, UCHAR *original_b) {

  point *horiz_least_energy_paths = malloc(width * sizeof(point));
  find_best_horiz_seam(width, height, batch_size, energy_r,
                       horiz_least_energy_paths, removed_mask);
  for (int x = 0; x < width; ++x) {
    int y = horiz_least_energy_paths[x].y;
    int index = x + y * width;

    if (removed_mask[index] == 0)
      removed_mask[index] = 2;
    else
      removed_mask[index] = 4;

    update_enery_around_pixel(x, y, energy_r, removed_mask, original_r,
                              original_g, original_b, width, height);
  }
  free(horiz_least_energy_paths);
}

void get_cropped_rgb(UCHAR *cropped_r, UCHAR *cropped_g, UCHAR *cropped_b,
                     UCHAR *original_r, UCHAR *original_g, UCHAR *original_b,
                     UCHAR *removed_mask, int width, int height, int n_cols,
                     int n_rows) {
  int cropped_pixel_count = (width) * (height);
  UCHAR *inter_r = malloc(cropped_pixel_count * sizeof(UCHAR));
  UCHAR *inter_g = malloc(cropped_pixel_count * sizeof(UCHAR));
  UCHAR *inter_b = malloc(cropped_pixel_count * sizeof(UCHAR));

  UCHAR *inter_mask = malloc(cropped_pixel_count * sizeof(UCHAR));

  int x, y;
  for (y = 0; y < height; ++y) {
    int reducer = 0;
    for (x = 0; x < width; ++x) {
      int index = x + y * width;
      if (removed_mask[index] == 1) {
        reducer++;
        continue;
      }
      int cropped_index = x - reducer + y * (width - n_cols);
      inter_mask[cropped_index] = removed_mask[index];

      inter_r[cropped_index] = original_r[index];
      inter_g[cropped_index] = original_g[index];
      inter_b[cropped_index] = original_b[index];
    }
  }
  for (x = 0; x < (width - n_cols); ++x) {
    int reducer = 0;
    for (y = 0; y < height; ++y) {
      int index = x + y * (width - n_cols);
      if (inter_mask[index] != 0) {
        reducer++;
        continue;
      }
      int cropped_index = x + (y - reducer) * (width - n_cols);
      cropped_r[cropped_index] = inter_r[index];
      cropped_g[cropped_index] = inter_g[index];
      cropped_b[cropped_index] = inter_b[index];
    }
  }
}

int main() {
  printf("Program start.\n");
  srand(time(NULL));
  printf("Random seed set %d.\n", rand());

  BMP_GetError();
  // const char *inFileBase = "main";
  const char *inFileBase = "rain";
  char inFile[100];
  sprintf(inFile, "%s.bmp", inFileBase);
  char outColFile[100];
  char outEnergyFile[100];
  char outCroppedFile[100];
  sprintf(outColFile, "%s_out_col.bmp", inFileBase);
  sprintf(outEnergyFile, "%s_out_energy.bmp", inFileBase);
  sprintf(outCroppedFile, "%s_out_cropped.bmp", inFileBase);

  int vert_crop_percent = 10;
  int horiz_crop_percent = 10;
  int batch_size = 1000;

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
    }
  }

  UCHAR *energy_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *energy_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *energy_b = malloc(pixelCount * sizeof(UCHAR));

  copy_rgb(energy_r, energy_g, energy_b, original_r, original_g, original_b,
           width, height);

  printf("H/D memory allocation done.\n");

  printf("Data sent to device.\n");

  energy(energy_r, energy_g, energy_b, width, height);

  printf("Data received from device.\n");

  // int n_cols = width * vert_crop_percent / 100;
  // int n_rows = height * horiz_crop_percent / 100;
  int n_cols = 5;
  int n_rows = 5;

  for (int i = 0; i < n_cols; i++) {
    remove_vert_seam(width, height, batch_size, energy_r, removed_mask,
                     original_r, original_g, original_b);
  }

  for (int i = 0; i < n_rows; i++) {
    remove_horiz_seam(width, height, batch_size, energy_r, removed_mask,
                      original_r, original_g, original_b);
  }
  // int i = n_cols, j = n_rows;
  // while (i + j > 0) {
  //   int rng = rand() % 2;
  //   if (rng == 0 && i > 0) {
  //     remove_vert_seam(width, height, batch_size, energy_r, removed_mask,
  //                      original_r, original_g, original_b);
  //     i--;
  //   } else if (j > 0) {
  //     remove_horiz_seam(width, height, batch_size, energy_r, removed_mask,
  //                       original_r, original_g, original_b);
  //     j--;
  //   }
  // }

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      int index = x + y * width;
      if (removed_mask[index] == 1) {
        original_r[index] = 255;
        original_g[index] = 0;
        original_b[index] = 0;
      }
      if (removed_mask[index] == 2) {
        original_r[index] = 0;
        original_g[index] = 0;
        original_b[index] = 255;
      }
      if (removed_mask[index] == 3) {
        original_r[index] = 255;
        original_g[index] = 0;
        original_b[index] = 255;
      }
      if (removed_mask[index] == 4) {
        original_r[index] = 0;
        original_g[index] = 255;
        original_b[index] = 255;
      }
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

  UCHAR *cropped_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_b = malloc(pixelCount * sizeof(UCHAR));
  get_cropped_rgb(cropped_r, cropped_g, cropped_b, original_r, original_g,
                  original_b, removed_mask, width, height, n_cols, n_rows);

  BMP *bmpCropped = BMP_Create(width - n_cols, height - n_rows, 32);
  for (y = 0; y < height - n_rows; ++y) {
    for (x = 0; x < width - n_cols; ++x) {
      int index = x + y * (width - n_cols);
      BMP_SetPixelRGB(bmpCropped, x, y, *(cropped_r + index),
                      *(cropped_g + index), *(cropped_b + index));
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
