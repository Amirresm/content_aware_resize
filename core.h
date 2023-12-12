#include "utils.h"
#include <time.h>

int find_best_vertical_seam_sol2(int width, int height, int batch_size,
                                 UCHAR *energy_matrix, seam *best_seam,
                                 UCHAR *removed_mask) {
  int least_total_energy_disruption = 99999999;
  for (int batch = 0; batch < batch_size; batch++) {
    int i = 0;
    int y = 0;
    int x = rand() % (width - 10) + 5;
    int total_energy_disruption = 0;
    point *path = (point *)malloc(height * sizeof(point));

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
            introduced_energy = energy_matrix[child.x + child.y * width];
          }
          break;

        case 0: // center
          // x_child = center_x;
          child.x = center_c.x;
          child.y = center_c.y;
          if (child.x == 0 || child.x == width - 1 || y == height - 1) {
            introduced_energy = 10000;
          } else {
            introduced_energy = energy_matrix[child.x + child.y * width];
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
            introduced_energy = energy_matrix[child.x + child.y * width];
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
int remove_vert_seam_sol2(int width, int height, int batch_size,
                          UCHAR *energy_r, UCHAR *removed_mask,
                          UCHAR *original_r, UCHAR *original_g,
                          UCHAR *original_b, seam *removed_seam) {
  int energy_disruption = find_best_vertical_seam_sol2(
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
seam *remove_vert_seams_sol2(UCHAR *energy_r, UCHAR *removed_mask,
                             UCHAR *original_r, UCHAR *original_g,
                             UCHAR *original_b, int width, int height,
                             int n_cols, int n_rows, int batch_size) {

  seam *seams = (seam *)malloc((n_cols + n_rows) * sizeof(seam));
  int energy_disruption = 0;
  for (int i = 0; i < n_cols; i++) {
    int ed =
        remove_vert_seam_sol2(width, height, batch_size, energy_r, removed_mask,
                              original_r, original_g, original_b, &seams[i]);
    energy_disruption += ed * ed;
  }
  return seams;
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
    point *path = (point *)malloc(height * sizeof(point));

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
    point *path = (point *)malloc(width * sizeof(point));

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

seam *remove_vert_and_horiz_seams_inorder(UCHAR *energy_r, UCHAR *removed_mask,
                                          UCHAR *original_r, UCHAR *original_g,
                                          UCHAR *original_b, int width,
                                          int height, int n_cols, int n_rows,
                                          int batch_size) {

  seam *seams = (seam *)malloc((n_cols + n_rows) * sizeof(seam));
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
  seam *seams = (seam *)malloc((n_cols + n_rows) * sizeof(seam));
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

  UCHAR *energy_copy = (UCHAR *)malloc(width * height * sizeof(UCHAR));
  memcpy(energy_copy, energy_r, width * height * sizeof(UCHAR));

  UCHAR *removed_mask_copy = (UCHAR *)malloc(width * height * sizeof(UCHAR));
  memcpy(removed_mask_copy, removed_mask, width * height * sizeof(UCHAR));

  seam *seams_copy = (seam *)malloc(n_seams * sizeof(seam));
  memcpy(seams_copy, seams, n_seams * sizeof(seam));
  for (int i = 0; i < n_seams; ++i) {
    seams_copy[i].path = (point *)malloc(seams[i].length * sizeof(point));
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
  seam *seams = (seam *)malloc((n_cols + n_rows) * sizeof(seam));
  for (int i = 0; i < n_cols + n_rows; ++i) {
    seams[i].type = 0;
    seams[i].length = 0;
    seams[i].energy = 0;
    seams[i].path = (point *)malloc((width + height) * sizeof(point));
  }
  int energy_disruption =
      dnc1(energy_r, removed_mask, original_r, original_g, original_b, width,
           height, n_cols, n_rows, batch_size, seams, 0, n_cols + n_rows);
  return seams;
}

seam *seam_carving_dp(UCHAR *energy_r, UCHAR *removed_mask, UCHAR *original_r,
                      UCHAR *original_g, UCHAR *original_b, int width,
                      int height, int n_cols, int n_rows, int batch_size) {
  seam *dp_seams = (seam *)malloc((n_cols + n_rows) * sizeof(seam));
  // Initialize DP table
  int ***dp = (int ***)malloc((width + 1) * sizeof(int **));
  for (int i = 0; i <= width; ++i) {
    dp[i] = (int **)malloc((height + 1) * sizeof(int *));
    for (int j = 0; j <= height; ++j) {
      dp[i][j] = (int *)malloc((n_cols + 1) * sizeof(int));
      for (int k = 0; k <= n_cols; ++k) {
        dp[i][j][k] = 999999999; // Initializing with a large value
      }
    }
  }
  // printf(dp_seams[0].path == NULL ? "true" : "false");
  // Fill base cases
  for (int i = 0; i <= width; ++i) {
    dp[i][0][0] = 0;
  }
  for (int i = 0; i <= height; ++i) {
    dp[0][i][0] = 0;
  }

  for (int cols_removed = 1; cols_removed <= n_cols; ++cols_removed) {
    for (int rows_removed = 0; rows_removed <= n_rows; ++rows_removed) {
      for (int w = 1; w <= width; ++w) {
        for (int h = 1; h <= height; ++h) {
          int ed_left = remove_vert_seam(
              width, height, batch_size, energy_r, removed_mask, original_r,
              original_g, original_b, &dp_seams[cols_removed + rows_removed]);
          int ed_right = remove_horiz_seam(
              width, height, batch_size, energy_r, removed_mask, original_r,
              original_g, original_b, &dp_seams[cols_removed + rows_removed]);

          if (rows_removed > 0) {
            int ed1 = dp[w][h][cols_removed - 1];
            dp[w][h][cols_removed] = ed_left + ed1;
          }

          if (cols_removed > 0) {
            int ed2 = dp[w][h][cols_removed];
            dp[w][h][cols_removed] = ed_right + ed2;
          }
        }
      }
    }
  }
  int w = width;
  int h = height;
  int cols_removed = n_cols;
  int rows_removed = n_rows;

  while (w > 0 && h > 0 && cols_removed > 0 && rows_removed >= 0) {
    int ed_left = remove_vert_seam(
        width, height, batch_size, energy_r, removed_mask, original_r,
        original_g, original_b, &dp_seams[cols_removed + rows_removed]);
    int ed_right = remove_horiz_seam(
        width, height, batch_size, energy_r, removed_mask, original_r,
        original_g, original_b, &dp_seams[cols_removed + rows_removed]);

    if (rows_removed > 0 &&
        dp[w][h][cols_removed] == ed_left + dp[w][h][cols_removed - 1]) {
      // This means a vertical seam was removed
      // Mark or store the seam information
      rows_removed--;

      // Update width (w) and height (h) based on the removal decision
      for (int i = h; i < height; ++i) {
        energy_r[i * width + w] = energy_r[i * width + w + 1];
        removed_mask[i * width + w] = removed_mask[i * width + w + 1];
        original_r[i * width + w] = original_r[i * width + w + 1];
        original_g[i * width + w] = original_g[i * width + w + 1];
        original_b[i * width + w] = original_b[i * width + w + 1];
      }
      w--; // Decrease width after removal
    } else if (cols_removed > 0 && dp[w][h][cols_removed] ==
                                       ed_right + dp[w][h][cols_removed - 1]) {
      // This means a horizontal seam was removed
      // Mark or store the seam information
      cols_removed--;

      // Update width (w) and height (h) based on the removal decision
      for (int i = w; i < width; ++i) {
        energy_r[h * width + i] = energy_r[(h + 1) * width + i];
        removed_mask[h * width + i] = removed_mask[(h + 1) * width + i];
        original_r[h * width + i] = original_r[(h + 1) * width + i];
        original_g[h * width + i] = original_g[(h + 1) * width + i];
        original_b[h * width + i] = original_b[(h + 1) * width + i];
      }
      h--; // Decrease height after removal
    }
  }
  // Free memory
  for (int i = 0; i <= width; ++i) {
    for (int j = 0; j <= height; ++j) {
      free(dp[i][j]);
    }
    free(dp[i]);
  }
  free(dp);

  return dp_seams;
}