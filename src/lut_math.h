#pragma once
#include "lut.h"

static inline float lut_interp(const float* table, float x) {
    if (x <= LUT_X_MIN) return table[0];
    if (x >= LUT_X_MAX) return table[LUT_N - 1];

    float pos = (x - LUT_X_MIN) * (float)(LUT_N - 1) / (LUT_X_MAX - LUT_X_MIN);
    int idx = (int)pos;
    float frac = pos - (float)idx;

    return table[idx] + frac * (table[idx + 1] - table[idx]);
}

static inline float fast_sigmoid(float x) { return lut_interp(SIGMOID_LUT, x); }
static inline float fast_tanh(float x)    { return lut_interp(TANH_LUT, x); }
