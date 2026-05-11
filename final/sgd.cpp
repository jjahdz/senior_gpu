#include "sgd.h"

void sgd_update(float* param, float grad, float lr) {
    *param -= lr * grad;
}