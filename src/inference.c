#include <math.h>
#include <string.h>
#include "weights_float.h"
#include "model_api.h"
#define MAX_LEN 12

static void gru_step(const float* x, float* h,
                      const float* w_ih, const float* w_hh,
                      const float* b_ih, const float* b_hh) {
    float gi[3 * HIDDEN];
    float gh[3 * HIDDEN];
    for (int g = 0; g < 3 * HIDDEN; g++) {
        float sum_i = b_ih[g];
        for (int k = 0; k < EMB_DIM; k++) sum_i += w_ih[g * EMB_DIM + k] * x[k];
        gi[g] = sum_i;
        float sum_h = b_hh[g];
        for (int k = 0; k < HIDDEN; k++) sum_h += w_hh[g * HIDDEN + k] * h[k];
        gh[g] = sum_h;
    }
    float new_h[HIDDEN];
    for (int j = 0; j < HIDDEN; j++) {
        float r = 1.0f / (1.0f + expf(-(gi[j] + gh[j])));
        float z = 1.0f / (1.0f + expf(-(gi[HIDDEN + j] + gh[HIDDEN + j])));
        float n = tanhf(gi[2 * HIDDEN + j] + r * gh[2 * HIDDEN + j]);
        new_h[j] = (1.0f - z) * n + z * h[j];
    }
    memcpy(h, new_h, sizeof(new_h));
}

void model_forward(const char* text, const int ids[MAX_LEN], int length,
                    float action_logits[N_ACTIONS],
                    float target_logits[N_TARGETS], float* value_out) {
    if (length <= 0) length = 1;
    if (length > MAX_LEN) length = MAX_LEN;

    float h_fwd[HIDDEN];
    memset(h_fwd, 0, sizeof(h_fwd));
    for (int t = 0; t < length; t++) {
        const float* emb = &EMBED_W[ids[t] * EMB_DIM];
        gru_step(emb, h_fwd, GRU_W_IH, GRU_W_HH, GRU_B_IH, GRU_B_HH);
    }

    float h_bwd[HIDDEN];
    memset(h_bwd, 0, sizeof(h_bwd));
    for (int t = length - 1; t >= 0; t--) {
        const float* emb = &EMBED_W[ids[t] * EMB_DIM];
        gru_step(emb, h_bwd, GRU_W_IH_REV, GRU_W_HH_REV, GRU_B_IH_REV, GRU_B_HH_REV);
    }

    float h_cat[2 * HIDDEN];
    memcpy(h_cat, h_fwd, sizeof(h_fwd));
    memcpy(h_cat + HIDDEN, h_bwd, sizeof(h_bwd));

    for (int a = 0; a < N_ACTIONS; a++) {
        float sum = ACTION_B[a];
        for (int k = 0; k < 2 * HIDDEN; k++) sum += ACTION_W[a * 2 * HIDDEN + k] * h_cat[k];
        action_logits[a] = sum;
    }
    for (int a = 0; a < N_TARGETS; a++) {
        float sum = TARGET_B[a];
        for (int k = 0; k < 2 * HIDDEN; k++) sum += TARGET_W[a * 2 * HIDDEN + k] * h_cat[k];
        target_logits[a] = sum;
    }
    float num_value;
    int num_present;
    extract_number(text, &num_value, &num_present);
    float num_feat = num_value / 180.0f;

    float v = VALUE_B[0];
    for (int k = 0; k < 2 * HIDDEN; k++) v += VALUE_W[k] * h_cat[k];
    v += VALUE_W[2 * HIDDEN + 0] * num_feat;
    v += VALUE_W[2 * HIDDEN + 1] * (float)num_present;
    *value_out = 1.0f / (1.0f + expf(-v));
}
