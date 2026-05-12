#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "model.h"
#include "reduction.h"
#include "adamw.h"
#include "data_gen.h"
#include "cuda_check.h"
#include "load_bin_data.h"
#include "sgd.h"

const float SGD_LR = 0.03f;//0.0005f;

// nvcc train.cu model.cu reduction.cu adamw.cpp data_gen.cpp load_bin_data.cpp sgd.cpp -o train_sgd.exe

int main()
{
    int n = N;
    size_t bytes = n * sizeof(float);

    int batch_size = 256;
    int points_per_batch = batch_size;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate CPU memory
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_partial = (float*)malloc(num_blocks * sizeof(float));

    if (!h_x || !h_y || !h_partial) {
        printf("Host malloc failed.\n");
        return 1;
    }

    // Allocate GPU memory
    float *d_x;
    float *d_y;
    float *d_partial;
    float *d_grad_a;
    float *d_grad_b;
    float *d_grad_c;
    float *d_grad_d;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_d, bytes));

    // Dataset selection
    bool use_news_data = true;
    bool use_powerplant_data = false;

    if (use_news_data) {
        load_bin_data("news_x.bin", "news_y.bin", h_x, h_y, n);
    }
    else if (use_powerplant_data) {
        load_bin_data("powerplant_x.bin", "powerplant_y.bin", h_x, h_y, n);
    }
    else {
        data_gen(h_x, h_y, n);
    }

    // Copy input data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // Initialize parameters
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 0.0f;

    if (use_news_data) {
        printf("Training News Popularity: x=SentimentHeadline, y=Facebook | N=%d epochs=%d lr=%.6f\n\n",
               N, EPOCHS, SGD_LR);
    }
    else if (use_powerplant_data) {
        printf("Training Power Plant | N=%d epochs=%d lr=%.6f\n\n",
               N, EPOCHS, SGD_LR);
    }
    else {
        printf("Training: Y = %.1fx^3 + %.1fx^2 + %.1fx + %.1f | N=%d epochs=%d lr=%.6f\n\n",
               TRUE_A, TRUE_B, TRUE_C, TRUE_BIAS, N, EPOCHS, SGD_LR);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        for (int i = 0; i < n; i += points_per_batch)
        {
            // Correctly handle the last partial batch
            int current_batch_size = points_per_batch;

            if (i + current_batch_size > n) {
                current_batch_size = n - i;
            }

            int current_blocks = (current_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Compute gradient contributions for this batch
            gradient_descent<<<current_blocks, BLOCK_SIZE>>>(
                d_x + i,
                d_y + i,
                d_grad_a,
                d_grad_b,
                d_grad_c,
                d_grad_d,
                a,
                b,
                c,
                d,
                current_batch_size
            );
            CUDA_CHECK(cudaGetLastError());

            // Reduce grad_a
            reduce_sum<<<current_blocks, BLOCK_SIZE>>>(d_grad_a, d_partial, current_batch_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, current_blocks * sizeof(float), cudaMemcpyDeviceToHost));

            float grad_a = 0.0f;
            for (int k = 0; k < current_blocks; k++) {
                grad_a += h_partial[k];
            }
            grad_a /= current_batch_size;

            // Reduce grad_b
            reduce_sum<<<current_blocks, BLOCK_SIZE>>>(d_grad_b, d_partial, current_batch_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, current_blocks * sizeof(float), cudaMemcpyDeviceToHost));

            float grad_b = 0.0f;
            for (int k = 0; k < current_blocks; k++) {
                grad_b += h_partial[k];
            }
            grad_b /= current_batch_size;

            // Reduce grad_c
            reduce_sum<<<current_blocks, BLOCK_SIZE>>>(d_grad_c, d_partial, current_batch_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, current_blocks * sizeof(float), cudaMemcpyDeviceToHost));

            float grad_c = 0.0f;
            for (int k = 0; k < current_blocks; k++) {
                grad_c += h_partial[k];
            }
            grad_c /= current_batch_size;

            // Reduce grad_d
            reduce_sum<<<current_blocks, BLOCK_SIZE>>>(d_grad_d, d_partial, current_batch_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, current_blocks * sizeof(float), cudaMemcpyDeviceToHost));

            float grad_d = 0.0f;
            for (int k = 0; k < current_blocks; k++) {
                grad_d += h_partial[k];
            }
            grad_d /= current_batch_size;

            // SGD update
            sgd_update(&a, grad_a, SGD_LR);
            sgd_update(&b, grad_b, SGD_LR);
            sgd_update(&c, grad_c, SGD_LR);
            sgd_update(&d, grad_d, SGD_LR);
        }

        if (epoch % 50 == 0 || epoch == EPOCHS - 1)
        {
            printf("[Epoch %3d] a=%.5f b=%.5f c=%.5f d=%.5f\n", epoch, a, b, c, d);
        }

        // Optional MSE logging for convergence plots
        if (epoch % 10 == 0 || epoch == EPOCHS - 1)
        {
            float epoch_mse = 0.0f;

            for (int j = 0; j < n; j++) {
                float x = h_x[j];
                float pred = a * x * x * x + b * x * x + c * x + d;
                float diff = pred - h_y[j];
                epoch_mse += diff * diff;
            }

            epoch_mse /= n;

            if (use_powerplant_data) {
                const float POWERPLANT_Y_STD = 7.4521f;
                printf("MSE_LOG %d %.6f\n", epoch, epoch_mse * POWERPLANT_Y_STD * POWERPLANT_Y_STD);
            }
            else if (use_news_data) {
                const float NEWS_Y_STD = 620.1699218750f;
                printf("MSE_LOG %d %.6f\n", epoch, epoch_mse * NEWS_Y_STD * NEWS_Y_STD);
            }
            else {
                printf("MSE_LOG %d %.6f\n", epoch, epoch_mse);
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Training time: %.2f ms\n", ms);

    printf("\n---- Results ----\n");
    printf("Learned: a=%.5f b=%.5f c=%.5f d=%.5f\n", a, b, c, d);

    float mse = 0.0f;

    for (int i = 0; i < n; i++) {
        float x = h_x[i];
        float pred = a * x * x * x + b * x * x + c * x + d;
        float diff = pred - h_y[i];
        mse += diff * diff;
    }

    mse /= n;

    if (use_powerplant_data) {
        const float POWERPLANT_Y_STD = 7.4521f;

        printf("MSE (normalized):     %.6f\n", mse);
        printf("MSE (original units): %.6f\n", mse * POWERPLANT_Y_STD * POWERPLANT_Y_STD);
        printf("Ojha reference MSE:   15.095050\n");
    }
    else if (use_news_data) {
        const float NEWS_Y_STD = 620.1699218750f;

        printf("MSE (normalized):     %.6f\n", mse);
        printf("MSE (original units): %.6f\n", mse * NEWS_Y_STD * NEWS_Y_STD);
        printf("Dataset: News Popularity\n");
        printf("x = SentimentHeadline, y = Facebook\n");
    }
    else {
        printf("Error:   a=%.6f b=%.6f c=%.6f d=%.6f\n",
               fabsf(a - TRUE_A),
               fabsf(b - TRUE_B),
               fabsf(c - TRUE_C),
               fabsf(d - TRUE_BIAS));

        printf("Final MSE: %.6f\n", mse);
    }
float baseline_mse = 0.0f;

for (int i = 0; i < n; i++) {
    float diff = 0.0f - h_y[i];  // prediction is mean in normalized space
    baseline_mse += diff * diff;
}

baseline_mse /= n;

if (use_news_data) {
    const float NEWS_Y_STD = 620.1699218750f;
    printf("Baseline MSE normalized: %.6f\n", baseline_mse);
    printf("Baseline MSE original:   %.6f\n", baseline_mse * NEWS_Y_STD * NEWS_Y_STD);
}
    free(h_x);
    free(h_y);
    free(h_partial);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_grad_a));
    CUDA_CHECK(cudaFree(d_grad_b));
    CUDA_CHECK(cudaFree(d_grad_c));
    CUDA_CHECK(cudaFree(d_grad_d));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}