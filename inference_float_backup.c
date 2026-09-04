#include <math.h>
#include <string.h>
#include <stdint.h>
#include "weights.h"
#include "lut_math.h"

#define MAX_LEN 12

// Quantize a float buffer to int8 with a fresh per-call scale.
static void quantize_h(const float* h, int8_t* h_q, float* scale_out) {
    float maxabs = 1e-8f;
    for (int i = 0; i < HIDDEN; i++) {
        float a = fabsf(h[i]);
        if (a > maxabs) maxabs = a;
    }
    float scale = maxabs / 127.0f;
    for (int i = 0; i < HIDDEN; i++)
        h_q[i] = (int8_t)lroundf(h[i] / scale);
    *scale_out = scale;
}

// One GRU timestep. x_q: int8 embedding row [EMB_DIM] (already quantized via EMBED_W_SCALE).
// h: [HIDDEN] float32 (in/out) -- kept float for accuracy of the gate outputs.
static void gru_step(const int8_t* x_q, float* h) {
    int8_t h_q[HIDDEN];
    float h_scale;
    quantize_h(h, h_q, &h_scale);

    float gi[3 * HIDDEN];
    float gh[3 * HIDDEN];

    for (int g = 0; g < 3 * HIDDEN; g++) {
        int32_t acc_i = 0;
        const int8_t* w_row_i = &GRU_W_IH[g * EMB_DIM];
        for (int k = 0; k < EMB_DIM; k++)
            acc_i += (int32_t)w_row_i[k] * (int32_t)x_q[k];
        gi[g] = acc_i * (GRU_W_IH_SCALE * EMBED_W_SCALE) + GRU_B_IH[g];

        int32_t acc_h = 0;
        const int8_t* w_row_h = &GRU_W_HH[g * HIDDEN];
        for (int k = 0; k < HIDDEN; k++)
            acc_h += (int32_t)w_row_h[k] * (int32_t)h_q[k];
        gh[g] = acc_h * (GRU_W_HH_SCALE * h_scale) + GRU_B_HH[g];
    }

    float new_h[HIDDEN];
    for (int j = 0; j < HIDDEN; j++) {
        float r = fast_sigmoid(gi[j] + gh[j]);
        float z = fast_sigmoid(gi[HIDDEN + j] + gh[HIDDEN + j]);
        float n = fast_tanh(gi[2 * HIDDEN + j] + r * gh[2 * HIDDEN + j]);
        new_h[j] = (1.0f - z) * n + z * h[j];
    }
    memcpy(h, new_h, sizeof(new_h));
}

void model_forward(const int ids[MAX_LEN], float action_logits[N_ACTIONS],
                    float target_logits[N_TARGETS], float* value_out) {
    float h[HIDDEN];
    memset(h, 0, sizeof(h));

    for (int t = 0; t < MAX_LEN; t++) {
        int id = ids[t];
        const int8_t* emb_row = &EMBED_W[id * EMB_DIM];
        gru_step(emb_row, h);
    }

    for (int a = 0; a < N_ACTIONS; a++) {
        float sum = ACTION_B[a];
        for (int k = 0; k < HIDDEN; k++) sum += ACTION_W[a * HIDDEN + k] * h[k];
        action_logits[a] = sum;
    }
    for (int a = 0; a < N_TARGETS; a++) {
        float sum = TARGET_B[a];
        for (int k = 0; k < HIDDEN; k++) sum += TARGET_W[a * HIDDEN + k] * h[k];
        target_logits[a] = sum;
    }
    float v = VALUE_B[0];
    for (int k = 0; k < HIDDEN; k++) v += VALUE_W[k] * h[k];
    *value_out = fast_sigmoid(v);
}
