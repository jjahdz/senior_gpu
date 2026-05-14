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
