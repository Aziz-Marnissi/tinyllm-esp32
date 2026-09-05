#ifndef MODEL_API_H
#define MODEL_API_H
#define MAX_LEN 12
#define N_ACTIONS 6
#define N_TARGETS 5
#ifdef __cplusplus
extern "C" {
#endif
int tokenize(const char* text, int out_ids[MAX_LEN]);
void extract_number(const char* text, float* out_value, int* out_present);
void model_forward(const int ids[MAX_LEN], int length, float num_feat, float num_present, float action_logits[N_ACTIONS],
                    float target_logits[N_TARGETS], float* value_out);
#ifdef __cplusplus
}
#endif
#endif
