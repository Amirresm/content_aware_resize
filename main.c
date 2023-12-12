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

typedef struct {
  point *path;
  char type;
  int energy;
  int length;
} seam;

void print_seam(seam *s) {
  printf("Seam type: %d, energy: %d, length: %d\n", s->type, s->energy,
         s->length);
  // for (int i = 0; i < s->length; ++i) {
  //   printf("(%d, %d) ", s->path[i].x, s->path[i].y);
  // }
  printf("\n");
}

void print_seams(seam *seams, int n_cols, int n_rows) {
  for (int i = 0; i < n_cols + n_rows; ++i) {
    print_seam(&seams[i]);
  }
}

int get_index(int x, int y, int width) { return x + y * width; }

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
  int index = get_index(x, y, width);
  if (removed_mask[index] == 0) {
    return;
  }
  if (x <= 0 || x >= width - 1) {
    return;
  }
  if (removed_mask[index] == 1) {
    if (dir == 0) {
      p->x += 1;
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
  int index = get_index(x, y, width);
  if (removed_mask[index] == 0) {
    return;
  }
  if (y <= 0 || y >= height - 1) {
    return;
  }
  if (removed_mask[index] == 2) {
    if (dir == 0) {
      p->y += 1;
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
int find_best_vertical_seam_sol2(int width, int height, int batch_size,
                                 UCHAR *energy_matrix,
                                 point *least_energy_paths,
                                 UCHAR *removed_mask) {
  int least_total_energy = 99999999;
  for (int batch = 0; batch < batch_size; batch++) {
    int y = 0;
    int x = rand() % width;
    // int x = batch / batch_size * width;
    int total_energy = 0;
    point *path = malloc(height * sizeof(point));

    while (y < height) {
      int index = get_index(x, y, width);
      path[y].x = x;
      path[y].y = y;
      int min_energy = 9999999;
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
            introduced_energy = energy_matrix[child.x + child.y * width];
          }
          break;

        case 0: // center
          // x_child = center_x;
          child.x = center_c.x;
          child.y = center_c.y;
          if (child.x == 0 || child.x == width - 1 || y == height - 1) {
            introduced_energy = 255;
          } else {
            introduced_energy = energy_matrix[child.x + child.y * width];
          }
          break;

        case 1: // right
          // x_child = right_x;
          child.x = right_c.x;
          child.y = right_c.y;
          if (child.x >= width - 2 || y == height - 1) {
            introduced_energy = 255;
          } else {
            introduced_energy = energy_matrix[child.x + child.y * width];
          }
          break;
        }
        if (introduced_energy < min_energy) {
          min_energy = introduced_energy;
          x = child.x;
          y = child.y;
        }
      }

      total_energy += min_energy;
    }
    if (total_energy < least_total_energy) {
      least_total_energy = total_energy;
      memcpy(least_energy_paths, path, height * sizeof(point));
    }
  }
  return least_total_energy;
}
int find_best_vertical_seam(int width, int height, int batch_size,
                            UCHAR *energy_matrix, seam *best_seam,
                            UCHAR *removed_mask) {
  int least_total_energy_disruption = 99999999;
  for (int batch = 0; batch < batch_size; batch++) {
    int i = 0;
    int y = 0;
    int x = rand() % (width - 10) + 5;
    int total_energy_disruption = 0;
    point *path = malloc(height * sizeof(point));

    while (y < height) {
      int index = get_index(x, y, width);
      path[i].x = x;
      path[i].y = y;
      i++;
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
          if (child.x < 5)
            continue;
          if (child.x <= 1 || y == height - 1) {
            introduced_energy = 10000;
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
            introduced_energy = 10000;
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
          if (child.x > width - 6)
            continue;
          if (child.x >= width - 2 || y == height - 1) {
            introduced_energy = 10000;
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
          y = child.y;
        }
      }

      total_energy_disruption += min_energy_disrupt;
    }
    if (i < height) {
      path[i].x = -1;
      path[i].y = -1;
    }
    if (total_energy_disruption < least_total_energy_disruption) {
      least_total_energy_disruption = total_energy_disruption;
      if (batch > 0)
        free(best_seam->path);
      best_seam->path = path;
      best_seam->energy = total_energy_disruption;
      best_seam->length = i;
      best_seam->type = 0;
    }
  }
  return least_total_energy_disruption;
}
int find_best_horiz_seam(int width, int height, int batch_size,
                         UCHAR *energy_matrix, seam *best_seam,
                         UCHAR *removed_mask) {
  int least_total_energy_disruption = 99999999;
  for (int batch = 0; batch < batch_size; batch++) {
    int i = 0;
    int x = 0;
    int y = rand() % (height - 10) + 5;
    int total_energy_disruption = 0;
    point *path = malloc(width * sizeof(point));

    while (x < width) {
      int index = get_index(x, y, width);
      path[i].x = x;
      path[i].y = y;
      i++;
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
        int introduced_energy = 10000;
        switch (random_child_index) {
        case -1: // up
          // y_child = up_y;
          child.x = up_c.x;
          child.y = up_c.y;
          if (child.y < 5)
            continue;
          if (child.y <= 1 || x == width - 1) {
            introduced_energy = 10000;
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
            introduced_energy = 10000;
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
          if (child.y > height - 6)
            continue;
          if (child.y >= height - 2 || x == width - 1) {
            introduced_energy = 10000;
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
          x = child.x;
          y = child.y;
        }
      }

      total_energy_disruption += min_energy_disrupt;
    }
    if (i < width) {
      path[i].x = -1;
      path[i].y = -1;
    }
    if (total_energy_disruption < least_total_energy_disruption) {
      least_total_energy_disruption = total_energy_disruption;

      if (batch > 0)
        free(best_seam->path);
      best_seam->path = path;
      best_seam->energy = total_energy_disruption;
      best_seam->length = i;
      best_seam->type = 1;
    }
  }
  return least_total_energy_disruption;
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

void update_energy_around_pixel(int x, int y, UCHAR *energy_matrix,
                                UCHAR *removed_mask, UCHAR *r, UCHAR *g,
                                UCHAR *b, int width, int height) {
  point left = {.x = x - 1 > 0 ? x - 1 : 1, .y = y};
  find_horiz_of_unremoved(&left, 0, removed_mask, width, height);
  point right = {.x = x + 1 < width - 1 ? x + 1 : width - 2, .y = y};
  find_horiz_of_unremoved(&right, 1, removed_mask, width, height);
  point up = {.x = x, .y = y - 1 > 0 ? y - 1 : 1};
  find_vert_of_unremoved(&up, 0, removed_mask, width, height);
  point down = {.x = x, .y = y + 1 < height - 1 ? y + 1 : height - 2};
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

int remove_vert_seam_sol2(int width, int height, int batch_size,
                          UCHAR *energy_r, UCHAR *removed_mask,
                          UCHAR *original_r, UCHAR *original_g,
                          UCHAR *original_b) {
  point *vert_least_energy_paths = malloc(height * sizeof(point));
  int energy =
      find_best_vertical_seam_sol2(width, height, batch_size, energy_r,
                                   vert_least_energy_paths, removed_mask);
  for (int y = 0; y < height; ++y) {
    int x = vert_least_energy_paths[y].x;
    int index = get_index(x, y, width);

    if (removed_mask[index] == 0)
      removed_mask[index] = 1;
    else
      removed_mask[index] = 3;

    update_energy_around_pixel(x, y, energy_r, removed_mask, original_r,
                               original_g, original_b, width, height);
  }
  free(vert_least_energy_paths);
  return energy;
}

int remove_vert_seam(int width, int height, int batch_size, UCHAR *energy_r,
                     UCHAR *removed_mask, UCHAR *original_r, UCHAR *original_g,
                     UCHAR *original_b, seam *removed_seam) {
  int energy_disruption = find_best_vertical_seam(
      width, height, batch_size, energy_r, removed_seam, removed_mask);
  for (int i = 0; i < removed_seam->length; ++i) {
    int x = removed_seam->path[i].x;
    int y = removed_seam->path[i].y;
    int index = get_index(x, y, width);

    if (removed_mask[index] == 0)
      removed_mask[index] = 1;
    else
      removed_mask[index] = 3;

    update_energy_around_pixel(x, y, energy_r, removed_mask, original_r,
                               original_g, original_b, width, height);
  }
  return energy_disruption;
}

int remove_horiz_seam(int width, int height, int batch_size, UCHAR *energy_r,
                      UCHAR *removed_mask, UCHAR *original_r, UCHAR *original_g,
                      UCHAR *original_b, seam *removed_seam) {
  int energy_disruption = find_best_horiz_seam(
      width, height, batch_size, energy_r, removed_seam, removed_mask);
  for (int i = 0; i < removed_seam->length; ++i) {
    int x = removed_seam->path[i].x;
    int y = removed_seam->path[i].y;
    int index = get_index(x, y, width);

    if (removed_mask[index] == 0)
      removed_mask[index] = 2;
    else
      removed_mask[index] = 4;

    update_energy_around_pixel(x, y, energy_r, removed_mask, original_r,
                               original_g, original_b, width, height);
  }
  return energy_disruption;
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
      int index = get_index(x, y, width);
      if (removed_mask[index] == 1) {
        reducer++;
        continue;
      }
      int cropped_index = get_index(x - reducer, y, width - n_cols);
      inter_mask[cropped_index] = removed_mask[index];

      inter_r[cropped_index] = original_r[index];
      inter_g[cropped_index] = original_g[index];
      inter_b[cropped_index] = original_b[index];
    }
  }
  for (x = 0; x < (width - n_cols); ++x) {
    int reducer = 0;
    for (y = 0; y < height; ++y) {
      int index = get_index(x, y, width - n_cols);
      if (inter_mask[index] == 2) {
        reducer++;
        continue;
      }
      int cropped_index = get_index(x, y - reducer, width - n_cols);
      cropped_r[cropped_index] = inter_r[index];
      cropped_g[cropped_index] = inter_g[index];
      cropped_b[cropped_index] = inter_b[index];
    }
  }
}
void get_cropped_rgb2(UCHAR *cropped_r, UCHAR *cropped_g, UCHAR *cropped_b,
                      UCHAR *original_r, UCHAR *original_g, UCHAR *original_b,
                      UCHAR *removed_mask, int width, int height, int n_cols,
                      int n_rows) {

  int index = 0, cropped_index = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      index = get_index(x, y, width);
      if (removed_mask[index] != 1) {
        cropped_r[cropped_index] = original_r[index];
        cropped_g[cropped_index] = original_g[index];
        cropped_b[cropped_index] = original_b[index];
        cropped_index++;
      }
    }
  }
}

seam *remove_vert_and_horiz_seams_inorder(UCHAR *energy_r, UCHAR *removed_mask,
                                          UCHAR *original_r, UCHAR *original_g,
                                          UCHAR *original_b, int width,
                                          int height, int n_cols, int n_rows,
                                          int batch_size) {

  seam *seams = malloc((n_cols + n_rows) * sizeof(seam));
  int energy_disruption = 0;
  for (int i = 0; i < n_cols; i++) {
    int ed = remove_vert_seam(width, height, batch_size, energy_r, removed_mask,
                              original_r, original_g, original_b, &seams[i]);
    energy_disruption += ed * ed;
  }

  for (int i = 0; i < n_rows; i++) {
    int ed = remove_horiz_seam(width, height, batch_size, energy_r,
                               removed_mask, original_r, original_g, original_b,
                               &seams[n_cols + i]);
    energy_disruption += ed * ed;
  }
  return seams;
}
seam *remove_vert_and_horiz_seams_randomly(UCHAR *energy_r, UCHAR *removed_mask,
                                           UCHAR *original_r, UCHAR *original_g,
                                           UCHAR *original_b, int width,
                                           int height, int n_cols, int n_rows,
                                           int batch_size) {
  seam *seams = malloc((n_cols + n_rows) * sizeof(seam));
  srand(time(NULL));
  int energy_disruption = 0;
  int i = n_cols, j = n_rows;
  int k = 0;
  while (i + j > 0) {
    int rng = rand() % 2;
    if ((rng == 0 && i > 0) || (j == 0)) {
      int ed =
          remove_vert_seam(width, height, batch_size, energy_r, removed_mask,
                           original_r, original_g, original_b, &seams[k]);
      energy_disruption += ed * ed;
      i--;
    } else if (j > 0) {
      int ed =
          remove_horiz_seam(width, height, batch_size, energy_r, removed_mask,
                            original_r, original_g, original_b, &seams[k]);
      energy_disruption += ed * ed;
      j--;
    }
    k++;
  }
  return seams;
}

int dnc1(UCHAR *energy_r, UCHAR *removed_mask, UCHAR *original_r,
         UCHAR *original_g, UCHAR *original_b, int width, int height,
         int n_cols, int n_rows, int batch_size, seam *seams, int current_seam,
         int n_seams) {
  if (n_cols == 0 && n_rows == 0) {
    return 0;
  } else if (n_cols == 0) {
    int ed_left = remove_horiz_seam(width, height, batch_size, energy_r,
                                    removed_mask, original_r, original_g,
                                    original_b, &seams[current_seam]);
    int ed1 = dnc1(energy_r, removed_mask, original_r, original_g, original_b,
                   width, height, n_cols, n_rows - 1, batch_size, seams,
                   current_seam + 1, n_seams);
    return ed_left + ed1;
  } else if (n_rows == 0) {
    int ed_right = remove_vert_seam(width, height, batch_size, energy_r,
                                    removed_mask, original_r, original_g,
                                    original_b, &seams[current_seam]);
    int ed2 = dnc1(energy_r, removed_mask, original_r, original_g, original_b,
                   width, height, n_cols - 1, n_rows, batch_size, seams,
                   current_seam + 1, n_seams);
    return ed_right + ed2;
  }

  UCHAR *energy_copy = malloc(width * height * sizeof(UCHAR));
  memcpy(energy_copy, energy_r, width * height * sizeof(UCHAR));

  UCHAR *removed_mask_copy = malloc(width * height * sizeof(UCHAR));
  memcpy(removed_mask_copy, removed_mask, width * height * sizeof(UCHAR));

  seam *seams_copy = malloc(n_seams * sizeof(seam));
  memcpy(seams_copy, seams, n_seams * sizeof(seam));
  for (int i = 0; i < n_seams; ++i) {
    seams_copy[i].path = malloc(seams[i].length * sizeof(point));
    memcpy(seams_copy[i].path, seams[i].path, seams[i].length * sizeof(point));
  }

  int ed_left = remove_vert_seam(width, height, batch_size, energy_r,
                                 removed_mask, original_r, original_g,
                                 original_b, &seams[current_seam]);

  int ed_right = remove_horiz_seam(width, height, batch_size, energy_copy,
                                   removed_mask_copy, original_r, original_g,
                                   original_b, &seams_copy[current_seam]);

  int ed1 = dnc1(energy_r, removed_mask, original_r, original_g, original_b,
                 width, height, n_cols - 1, n_rows, batch_size, seams,
                 current_seam + 1, n_seams);
  int ed2 = dnc1(energy_copy, removed_mask_copy, original_r, original_g,
                 original_b, width, height, n_cols, n_rows - 1, batch_size,
                 seams_copy, current_seam + 1, n_seams);

  if (ed1 + ed_left < ed2 + ed_right) {
    return ed1 + ed_left;
  } else {
    memcpy(energy_r, energy_copy, width * height * sizeof(UCHAR));
    memcpy(removed_mask, removed_mask_copy, width * height * sizeof(UCHAR));
    memcpy(seams, seams_copy, (n_cols + n_rows) * sizeof(seam));
    for (int i = 0; i < n_cols + n_rows; ++i) {
      memcpy(seams[i].path, seams_copy[i].path,
             seams_copy[i].length * sizeof(point));
    }
    return ed2 + ed_right;
  }
}

seam *remove_vert_and_horiz_seams_dnc(UCHAR *energy_r, UCHAR *removed_mask,
                                      UCHAR *original_r, UCHAR *original_g,
                                      UCHAR *original_b, int width, int height,
                                      int n_cols, int n_rows, int batch_size) {
  seam *seams = malloc((n_cols + n_rows) * sizeof(seam));
  for (int i = 0; i < n_cols + n_rows; ++i) {
    seams[i].type = 0;
    seams[i].length = 0;
    seams[i].energy = 0;
    seams[i].path = malloc((width + height) * sizeof(point));
  }
  int energy_disruption =
      dnc1(energy_r, removed_mask, original_r, original_g, original_b, width,
           height, n_cols, n_rows, batch_size, seams, 0, n_cols + n_rows);
  return seams;
}

void remove_one_seam_from_image(UCHAR *dest_r, UCHAR *dest_g, UCHAR *dest_b,
                                UCHAR *src_r, UCHAR *src_g, UCHAR *src_b,
                                seam *current_seam, int width, int height,
                                seam *other_seams, int current_seam_index,
                                int total_seam_count) {
  printf("remove_one_seam_from_image for s %d: \n", current_seam_index);
  print_seam(current_seam);
  if (current_seam->type == 0) {
    for (int i = 0; i < current_seam->length; ++i) {
      int y = current_seam->path[i].y;
      for (int x = 0; x < width; ++x) {
        int target_x = current_seam->path[i].x;
        int index = get_index(x, y, width);
        int dest_x = x;
        if (x == target_x) {
          continue;
        } else if (x > target_x) {
          dest_x = x - 1;
        } else {
          dest_x = x;
        }
        int dest_index = get_index(dest_x, y, width - 1);
        dest_r[dest_index] = src_r[index];
        dest_g[dest_index] = src_g[index];
        dest_b[dest_index] = src_b[index];
      }
    }

    for (int i = 0; i < current_seam->length; ++i) {
      int x = current_seam->path[i].x;
      int y = current_seam->path[i].y;
      for (int j = current_seam_index + 1; j < total_seam_count; ++j) {
        for (int k = 0; k < other_seams[j].length; ++k) {
          if (other_seams[j].path[k].x > x && other_seams[j].path[k].y == y) {
            other_seams[j].path[k].x -= 1;
          }
        }
      }
    }
  } else {
    for (int i = 0; i < current_seam->length; ++i) {
      int x = current_seam->path[i].x;
      for (int y = 0; y < height; ++y) {
        int target_y = current_seam->path[i].y;
        int index = get_index(x, y, width);
        int dest_y = y;
        if (y == target_y) {
          continue;
        } else if (y > target_y) {
          dest_y = y - 1;
        } else {
          dest_y = y;
        }
        int dest_index = get_index(x, dest_y, width);
        dest_r[dest_index] = src_r[index];
        dest_g[dest_index] = src_g[index];
        dest_b[dest_index] = src_b[index];
      }
    }

    for (int i = 0; i < current_seam->length; ++i) {
      int x = current_seam->path[i].x;
      int y = current_seam->path[i].y;
      for (int j = current_seam_index + 1; j < total_seam_count; ++j) {
        for (int k = 0; k < other_seams[j].length; ++k) {
          if (other_seams[j].path[k].y > y && other_seams[j].path[k].x == x) {
            other_seams[j].path[k].y -= 1;
          }
        }
      }
    }
  }
}

int main(int argc, char const *argv[]) {
  // parse args
  if (argc < 2) {
    printf("Usage: %s <solution> <vertical_resize> <horizental_resize> "
           "<batch_size> <file_name> ...\n",
           argv[0]);
    return 1;
  }
  int solution = 4;
  int vert_crop_percent = 25;
  int horiz_crop_percent = 10;

  int batch_size = 100;
  // const char *inFileBase = "main";
  const char *inFileBase = "rain";
  const char *in_path = "";
  const char *out_path = "";

  if (argc == 2) {
    solution = atoi(argv[1]);
  }
  if (argc == 3) {
    solution = atoi(argv[1]);
    vert_crop_percent = atoi(argv[2]);
  }
  if (argc == 4) {
    solution = atoi(argv[1]);
    vert_crop_percent = atoi(argv[2]);
    horiz_crop_percent = atoi(argv[3]);
  }
  if (argc == 5) {
    solution = atoi(argv[1]);
    vert_crop_percent = atoi(argv[2]);
    horiz_crop_percent = atoi(argv[3]);
    batch_size = atoi(argv[4]);
  }
  if (argc == 6) {
    solution = atoi(argv[1]);
    vert_crop_percent = atoi(argv[2]);
    horiz_crop_percent = atoi(argv[3]);
    batch_size = atoi(argv[4]);
    inFileBase = argv[5];
  }
  if (argc == 7) {
    solution = atoi(argv[1]);
    vert_crop_percent = atoi(argv[2]);
    horiz_crop_percent = atoi(argv[3]);
    batch_size = atoi(argv[4]);
    inFileBase = argv[5];
    in_path = argv[6];
  }
  if (argc == 8) {
    solution = atoi(argv[1]);
    vert_crop_percent = atoi(argv[2]);
    horiz_crop_percent = atoi(argv[3]);
    batch_size = atoi(argv[4]);
    inFileBase = argv[5];
    in_path = argv[6];
    out_path = argv[7];
  }
  srand(time(NULL));

  BMP_GetError();
  char inFile[100];
  sprintf(inFile, "%s%s.bmp", in_path, inFileBase);
  char outColFile[100];
  char outEnergyFile[100];
  char outCroppedFile[100];
  sprintf(outColFile, "%s%s_out_col.bmp", out_path, inFileBase);
  sprintf(outEnergyFile, "%s%s_out_energy.bmp", out_path, inFileBase);
  sprintf(outCroppedFile, "%s%s_out_cropped.bmp", out_path, inFileBase);

  UINT width, height;
  UINT x, y;
  BMP *bmp;

  printf("Reading %s ...\n", inFile);
  bmp = BMP_ReadFile(inFile);
  BMP_CHECK_ERROR(stdout, -1);

  printf("Image loaded.\n");

  width = BMP_GetWidth(bmp);
  height = BMP_GetHeight(bmp);
  unsigned int pixelCount = width * height;
  BMP_CHECK_ERROR(stderr, -2);
  // int n_cols = width * vert_crop_percent / 100;
  // int n_rows = solution == 4 ? height * horiz_crop_percent / 100 : 0;
  int n_cols = 10;
  int n_rows = 10;

  printf("Running solution %d with barch size %d \n", solution, batch_size);
  printf("Removing %d vertical seams and %d horizontal seams\n", n_cols,
         n_rows);

  printf("Width: %d, Height: %d\n", (int)width, (int)height);

  clock_t begin = clock();

  UCHAR *reference_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *reference_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *reference_b = malloc(pixelCount * sizeof(UCHAR));

  UCHAR *original_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *original_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *original_b = malloc(pixelCount * sizeof(UCHAR));

  UCHAR *removed_mask = malloc(pixelCount * sizeof(UCHAR));
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      int index = get_index(x, y, width);
      BMP_GetPixelRGB(bmp, x, y, reference_r + index, reference_g + index,
                      reference_b + index);
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

  energy(energy_r, energy_g, energy_b, width, height);

  seam *removed_seams;

  if (solution == 2) {
    printf("Running Solution 2...\n");
    int total_energy = 0;
    for (int i = 0; i < n_cols; i++) {
      int ed = remove_vert_seam_sol2(width, height, batch_size, energy_r,
                                     removed_mask, original_r, original_g,
                                     original_b);
      total_energy += ed;
    }
    printf("Total energy: %d\n", total_energy);
  } else if (solution == 3) {
    printf("Running Solution 3...\n");
    int energy_disruption = 0;
    for (int i = 0; i < n_cols; i++) {
      int ed =
          remove_vert_seam(width, height, batch_size, energy_r, removed_mask,
                           original_r, original_g, original_b, NULL);
      energy_disruption += ed;
    }
    printf("Total energy: %d\n", energy_disruption);
  } else if (solution == 4) {
    printf("Running Solution 4...\n");
    removed_seams = remove_vert_and_horiz_seams_dnc(
        energy_r, removed_mask, original_r, original_g, original_b, width,
        height, n_cols, n_rows, batch_size);

    int energy_disruption = 0;
    for (int i = 0; i < n_cols + n_rows; i++) {
      energy_disruption += removed_seams[i].energy;
    }
    printf("Energy disruption: %d\n", energy_disruption);
  }

  // for (int x = 0; x < width; ++x)
  // {
  //   for (int y = 0; y < height; ++y)
  //   {
  //     int index = get_index(x, y, width);
  //     if (x == 0 && removed_mask[index] != 0)
  //     {
  //       removed_mask[index] = 0;
  //     }
  //     if (y == 0 && removed_mask[index] != 0)
  //     {
  //       removed_mask[index] = 0;
  //     }
  //     if (x == 1 && removed_mask[index] != 0)
  //     {
  //       removed_mask[y * width] = removed_mask[index];
  //     }
  //     if (y == 1 && removed_mask[index] != 0)
  //     {
  //       removed_mask[x] = removed_mask[index];
  //     }
  //   }
  // }
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      int index = get_index(x, y, width);
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

  // for (int i = 0; i < n_cols + n_rows; ++i) {
  //   for (int j = 0; j < removed_seams[i].length; ++j) {
  //     int x = removed_seams[i].path[j].x;
  //     int y = removed_seams[i].path[j].y;
  //     int index = get_index(x, y, width);
  //     original_r[index] = 255;
  //     original_g[index] = i * 255 / (n_cols + n_rows);
  //     original_b[index] = 0;
  //   }
  // }

  UCHAR *cropped_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_b = malloc(pixelCount * sizeof(UCHAR));

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("EXECTION TIME: %f\n", time_spent);

  BMP *bmpEnergy = BMP_Create(width, height, 32);
  BMP *bmpOut = BMP_Create(width, height, 32);
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      int index = get_index(x, y, width);
      BMP_SetPixelRGB(bmpEnergy, x, y, *(energy_r + index), *(energy_g + index),
                      *(energy_b + index));
      BMP_SetPixelRGB(bmpOut, x, y, *(original_r + index),
                      *(original_g + index), *(original_b + index));
    }
  }

  int cropped_width = width;
  int cropped_height = height;
  for (int i = 0; i < n_cols + n_rows; i++) {
    remove_one_seam_from_image(cropped_r, cropped_g, cropped_b, original_r,
                               original_g, original_b, &removed_seams[i],
                               cropped_width, cropped_height, removed_seams, i,
                               n_cols + n_rows);
    copy_rgb(original_r, original_g, original_b, cropped_r, cropped_g,
             cropped_b, cropped_width, cropped_height);
    if (removed_seams[i].type == 0) {
      cropped_width--;
    } else {
      cropped_height--;
    }
  }
  print_seams(removed_seams, n_cols, n_rows);

  // for (int i = 0; i < n_cols + n_rows; ++i) {
  //   for (int j = 0; j < removed_seams[i].length; ++j) {
  //     int x = removed_seams[i].path[j].x;
  //     int y = removed_seams[i].path[j].y;
  //     int index = get_index(x, y, cropped_width);
  //     cropped_r[index] = 255;
  //     cropped_g[index] = (i)*255 / (n_cols + n_rows);
  //     cropped_b[index] = 0;
  //   }
  // }

  BMP *bmpCropped = BMP_Create(cropped_width, cropped_height, 32);
  for (y = 0; y < height; ++y) {
    for (x = 0; x < cropped_width; ++x) {
      int index = get_index(x, y, cropped_width);
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
