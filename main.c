#include "core.h"
#include "cuda.cuh"
#include "qdbmp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
  // parse args
  if (argc < 2) {
    printf("Usage: %s <solution> <vertical_resize> <horizental_resize> "
           "<batch_size> <file_name> ...\n",
           argv[0]);
    return 1;
  }
  int solution = 4;
  int vert_crop_percent = 10;
  int horiz_crop_percent = 10;

  int batch_size = 50;
  int method = 3;
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
  if (argc == 9) {
    solution = atoi(argv[1]);
    vert_crop_percent = atoi(argv[2]);
    horiz_crop_percent = atoi(argv[3]);
    batch_size = atoi(argv[4]);
    inFileBase = argv[5];
    in_path = argv[6];
    out_path = argv[7];
    method = atoi(argv[8]);
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
  BMP *bmp;

  printf("Reading %s ...\n", inFile);
  bmp = BMP_ReadFile(inFile);
  BMP_CHECK_ERROR(stdout, -1);

  printf("Image loaded.\n");

  width = BMP_GetWidth(bmp);
  height = BMP_GetHeight(bmp);
  unsigned int pixelCount = width * height;
  BMP_CHECK_ERROR(stderr, -2);
  int n_cols = width * vert_crop_percent / 100;
  int n_rows = solution == 4 ? height * horiz_crop_percent / 100 : 0;

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
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int index = get_index(x, y, width);
      BMP_GetPixelRGB(bmp, x, y, original_r + index, original_g + index,
                      original_b + index);
      BMP_GetPixelRGB(bmp, x, y, reference_r + index, reference_g + index,
                      reference_b + index);

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
    removed_seams = remove_vert_seams_sol2(energy_r, removed_mask, original_r,
                                           original_g, original_b, width,
                                           height, n_cols, n_rows, batch_size);
    int total_energy = 0;
    for (int i = 0; i < n_cols + n_rows; i++) {
      total_energy += removed_seams[i].energy;
    }
    printf("Total energy: %d\n", total_energy);
  } else if (solution == 3) {
    printf("Running Solution 3...\n");
    removed_seams = remove_vert_and_horiz_seams_inorder(
        energy_r, removed_mask, original_r, original_g, original_b, width,
        height, n_cols, n_rows, batch_size);
    int energy_disruption = 0;
    for (int i = 0; i < n_cols + n_rows; i++) {
      energy_disruption += removed_seams[i].energy;
    }
    printf("Total energy: %d\n", energy_disruption);

  } else if (solution == 4) {
    printf("Running Solution 4...\n");
    if (method == 0) {
      removed_seams = remove_vert_and_horiz_seams_inorder(
          energy_r, removed_mask, original_r, original_g, original_b, width,
          height, n_cols, n_rows, batch_size);
    } else if (method == 1) {
      removed_seams = remove_vert_and_horiz_seams_randomly(
          energy_r, removed_mask, original_r, original_g, original_b, width,
          height, n_cols, n_rows, batch_size);
    } else if (method == 2) {
      removed_seams = remove_vert_and_horiz_seams_dnc(
          energy_r, removed_mask, original_r, original_g, original_b, width,
          height, n_cols, n_rows, batch_size);
    } else if (method == 3) {
      seam *result = seam_carving_dp(energy_r, removed_mask, original_r,
                                     original_g, original_b, width, height,
                                     n_cols, n_rows, batch_size);
    } else {
      printf("Invalid method\n");
      return 1;
    }

    int energy_disruption = 0;
    for (int i = 0; i < n_cols + n_rows; i++) {
      energy_disruption += removed_seams[i].energy;
    }
    printf("Energy disruption: %d\n", energy_disruption);
  }

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

  UCHAR *cropped_r = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_g = malloc(pixelCount * sizeof(UCHAR));
  UCHAR *cropped_b = malloc(pixelCount * sizeof(UCHAR));

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("EXECTION TIME: %f\n", time_spent);

  BMP *bmpEnergy = BMP_Create(width, height, 32);
  BMP *bmpOut = BMP_Create(width, height, 32);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int index = get_index(x, y, width);
      BMP_SetPixelRGB(bmpEnergy, x, y, *(energy_r + index), *(energy_g + index),
                      *(energy_b + index));
      BMP_SetPixelRGB(bmpOut, x, y, *(original_r + index),
                      *(original_g + index), *(original_b + index));
    }
  }

  clock_t beginSaving = clock();

  int cropped_width = width;
  int cropped_height = height;
  for (int i = 0; i < n_cols + n_rows; i++) {
    remove_one_seam_from_image(cropped_r, cropped_g, cropped_b, reference_r,
                               reference_g, reference_b, &removed_seams[i],
                               cropped_width, cropped_height, removed_seams, i,
                               n_cols + n_rows);
    copy_rgb(reference_r, reference_g, reference_b, cropped_r, cropped_g,
             cropped_b, cropped_width, cropped_height);
    if (removed_seams[i].type == 0) {
      cropped_width--;
    } else {
      cropped_height--;
    }
  }

  BMP *bmpCropped = BMP_Create(cropped_width, cropped_height, 32);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < cropped_width; ++x) {
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
  clock_t endSaving = clock();
  double time_saving = (double)(endSaving - beginSaving) / CLOCKS_PER_SEC;
  printf("SAVE TIME: %f\n", time_saving);
  double time_total = (double)(endSaving - begin) / CLOCKS_PER_SEC;
  printf("TOTAL TIME: %f\n", time_total);

  BMP_Free(bmp);

  return 0;
}
