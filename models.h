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
