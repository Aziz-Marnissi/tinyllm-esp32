#include <stdio.h>
#include "model_api.h"
static const char* ACTIONS[N_ACTIONS] = {"on", "off", "set_power", "set_degree", "set_timer", "check_status"};
static const char* TARGETS[N_TARGETS] = {"led", "motor_28BYJ48", "servo", "timer", "temp_sensor"};
int main() {
    const char* tests[] = {
        "turn on the led",
        "set servo to 90 degrees",
        "trun off motor",
        "brightness 45",
        "set a timer for 10 minutes",
        "what's the temperature",
        "is the led on",
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    for (int t = 0; t < n; t++) {
        int ids[MAX_LEN];
        int length = tokenize(tests[t], ids);
        float action_logits[N_ACTIONS], target_logits[N_TARGETS], value_out;
        model_forward(ids, length, action_logits, target_logits, &value_out);
        int best_a = 0, best_t = 0;
        for (int i = 1; i < N_ACTIONS; i++) if (action_logits[i] > action_logits[best_a]) best_a = i;
        for (int i = 1; i < N_TARGETS; i++) if (target_logits[i] > target_logits[best_t]) best_t = i;
        printf("\"%s\"\n", tests[t]);
        printf("  action=%s target=%s value=%.1f\n\n", ACTIONS[best_a], TARGETS[best_t], value_out * 180.0f);
    }
    return 0;
}
