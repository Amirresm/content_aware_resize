#include "models.h"
#include "qdbmp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void get_cropped_rgb(UCHAR *cropped_r, UCHAR *cropped_g, UCHAR *cropped_b,
                     UCHAR *original_r, UCHAR *original_g, UCHAR *original_b,
                     UCHAR *removed_mask, int width, int height, int n_cols,
                     int n_rows) {
  int cropped_pixel_count = (width) * (height);
  UCHAR *inter_r = (UCHAR *)malloc(cropped_pixel_count * sizeof(UCHAR));
  UCHAR *inter_g = (UCHAR *)malloc(cropped_pixel_count * sizeof(UCHAR));
  UCHAR *inter_b = (UCHAR *)malloc(cropped_pixel_count * sizeof(UCHAR));

  UCHAR *inter_mask = (UCHAR *)malloc(cropped_pixel_count * sizeof(UCHAR));

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

void remove_one_seam_from_image(UCHAR *dest_r, UCHAR *dest_g, UCHAR *dest_b,
                                UCHAR *src_r, UCHAR *src_g, UCHAR *src_b,
                                seam *current_seam, int width, int height,
                                seam *other_seams, int current_seam_index,
                                int total_seam_count) {
  // printf("remove_one_seam_from_image for s %d: \n", current_seam_index);
  // print_seam(current_seam);
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