#include "data_gen.h"
#include <stdlib.h>

void data_gen(float* h_x, float* h_y, int n)
{
    srand(42);
    for (int i = 0; i < n; i++)
    {
        float x  = ((float)i / n) * 2.0f - 1.0f;
        float x2 = x * x;
        float x3 = x2 * x;
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_x[i] = x;
        h_y[i] = TRUE_A * x3 + TRUE_B * x2 + TRUE_C * x + TRUE_BIAS + noise;
    }
}