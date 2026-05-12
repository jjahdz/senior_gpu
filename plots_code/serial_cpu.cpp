#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include "load_bin_data.h"
#include "adamw.h"
#include "data_gen.h"

int main() {
    const int n = 9568;
    const int epochs = 200;
    const float lr = 0.01f;
    const int batch_size = 256;

    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_y = (float*)malloc(n * sizeof(float));
    data_gen(h_x, h_y, n);

    float a=0, b=0, c=0, d=0;
    float m_a=0, v_a=0, m_b=0, v_b=0;
    float m_c=0, v_c=0, m_d=0, v_d=0;
    int timestep = 0;

    // Start timing
    auto t0 = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < n; i += batch_size) {
            int end = i + batch_size < n ? i + batch_size : n;
            float denom = (float)(end - i);
            float grad_a=0, grad_b=0, grad_c=0, grad_d=0;

            for (int j = i; j < end; j++) {
                float x = h_x[j];
                float x2 = x*x, x3 = x2*x;
                float err = (a*x3 + b*x2 + c*x + d) - h_y[j];
                grad_a += err * x3;
                grad_b += err * x2;
                grad_c += err * x;
                grad_d += err;
            }

            grad_a /= denom; grad_b /= denom;
            grad_c /= denom; grad_d /= denom;
            timestep++;

            adamw_update(&a, grad_a, &m_a, &v_a, 0.9f, 0.999f, 0.0f, lr, 1e-8f, timestep);
            adamw_update(&b, grad_b, &m_b, &v_b, 0.9f, 0.999f, 0.0f, lr, 1e-8f, timestep);
            adamw_update(&c, grad_c, &m_c, &v_c, 0.9f, 0.999f, 0.0f, lr, 1e-8f, timestep);
            adamw_update(&d, grad_d, &m_d, &v_d, 0.9f, 0.999f, 0.0f, lr, 1e-8f, timestep);
        }
    }

    // Stop timing
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Serial CPU training time: %.2f ms\n", ms);

    // MSE
    float mse = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = h_x[i];
        float pred = a*x*x*x + b*x*x + c*x + d;
        float diff = pred - h_y[i];
        mse += diff*diff;
    }
    mse /= n;
    float y_std = 7.4521f;
    printf("MSE (normalized):     %.6f\n", mse);
    printf("MSE (original units): %.6f\n", mse * y_std * y_std);
    printf("Learned: a=%.5f b=%.5f c=%.5f d=%.5f\n", a, b, c, d);

    free(h_x); free(h_y);
    return 0;
}
