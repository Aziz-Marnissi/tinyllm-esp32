// Same as eval_accuracy.c but prints every mismatch: text, true vs predicted.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model_api.h"

#define MAX_LINE 512

static const char* ACTIONS[6] = {"on", "off", "set_power", "set_degree", "set_timer", "check_status"};
static const char* TARGETS[5] = {"led", "motor_28BYJ48", "servo", "timer", "temp_sensor"};

static int find_index(const char* arr[], int n, const char* val) {
    for (int i = 0; i < n; i++) if (strcmp(arr[i], val) == 0) return i;
    return -1;
}

static int extract_str(const char* line, const char* key, char* out, int out_sz) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\": \"", key);
    const char* p = strstr(line, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    const char* end = strchr(p, '"');
    if (!end) return 0;
    int len = end - p;
    if (len >= out_sz) len = out_sz - 1;
    strncpy(out, p, len);
    out[len] = '\0';
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s val.jsonl\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    char line[MAX_LINE];
    int total = 0, correct_action = 0, correct_target = 0;

    while (fgets(line, sizeof(line), f)) {
        char text[MAX_LEN * 16], action_str[32], target_str[32];
        if (!extract_str(line, "text", text, sizeof(text))) continue;
        if (!extract_str(line, "action", action_str, sizeof(action_str))) continue;
        if (!extract_str(line, "target", target_str, sizeof(target_str))) continue;

        int true_action = find_index(ACTIONS, 6, action_str);
        int true_target = find_index(TARGETS, 5, target_str);
        if (true_action < 0 || true_target < 0) continue;

        int ids[MAX_LEN];
        int length = tokenize(text, ids);
        float num_raw, num_feat;
        int num_present;
        extract_number(text, &num_raw, &num_present);
        num_feat = num_raw / 180.0f;

        float action_logits[N_ACTIONS], target_logits[N_TARGETS], value_out;
        model_forward(ids, length, num_feat, (float)num_present, action_logits, target_logits, &value_out);

        int pred_action = 0;
        for (int i = 1; i < N_ACTIONS; i++)
            if (action_logits[i] > action_logits[pred_action]) pred_action = i;

        int pred_target = 0;
        for (int i = 1; i < N_TARGETS; i++)
            if (target_logits[i] > target_logits[pred_target]) pred_target = i;

        total++;
        int a_ok = (pred_action == true_action);
        int t_ok = (pred_target == true_target);
        if (a_ok) correct_action++;
        if (t_ok) correct_target++;

        if (!a_ok || !t_ok) {
            printf("MISMATCH: \"%s\"\n", text);
            printf("  true:  action=%s target=%s\n", ACTIONS[true_action], TARGETS[true_target]);
            printf("  pred:  action=%s target=%s\n", ACTIONS[pred_action], TARGETS[pred_target]);
        }
    }
    fclose(f);

    printf("\ntotal=%d action_acc=%.4f target_acc=%.4f\n",
           total, (double)correct_action / total, (double)correct_target / total);
    return 0;
}
