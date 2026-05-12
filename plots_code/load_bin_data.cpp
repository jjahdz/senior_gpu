#include <fstream>
#include "load_bin_data.h"
void load_bin_data(float* h_x, float* h_y, int n)
{
    FILE* fx = fopen("powerplant_x.bin", "rb");
    FILE* fy = fopen("powerplant_y.bin", "rb");
    fread(h_x, sizeof(float), n, fx);
    fread(h_y, sizeof(float), n, fy);
    fclose(fx);
    fclose(fy);
}
/*#include <cstdio>
#include <cstdlib>
#include "load_bin_data.h"

void load_bin_data(
    const char* x_file,
    const char* y_file,
    float* h_x,
    float* h_y,
    int n
)
{
    FILE* fx = fopen(x_file, "rb");
    FILE* fy = fopen(y_file, "rb");

    if (!fx) {
        printf("Error: could not open x file: %s\n", x_file);
        exit(1);
    }

    if (!fy) {
        printf("Error: could not open y file: %s\n", y_file);
        fclose(fx);
        exit(1);
    }

    size_t read_x = fread(h_x, sizeof(float), n, fx);
    size_t read_y = fread(h_y, sizeof(float), n, fy);

    fclose(fx);
    fclose(fy);

    if (read_x != (size_t)n) {
        printf("Error: expected %d x values, read %zu from %s\n", n, read_x, x_file);
        exit(1);
    }

    if (read_y != (size_t)n) {
        printf("Error: expected %d y values, read %zu from %s\n", n, read_y, y_file);
        exit(1);
    }

    printf("Loaded %d samples from:\n", n);
    printf("  x: %s\n", x_file);
    printf("  y: %s\n", y_file);
}*/