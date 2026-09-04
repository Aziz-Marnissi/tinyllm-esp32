#include <stdio.h>
#include "tokenizer.c"

int main() {
    const char* tests[] = {
        "turn on the led",
        "set servo to 90 degrees",
        "trun off motor",
        "brightness 45",
    };
    int n = sizeof(tests) / sizeof(tests[0]);

    for (int t = 0; t < n; t++) {
        int ids[MAX_LEN];
        tokenize(tests[t], ids);
        printf("\"%s\" -> [", tests[t]);
        for (int i = 0; i < MAX_LEN; i++) {
            printf("%d%s", ids[i], (i < MAX_LEN - 1) ? ", " : "");
        }
        printf("]\n");
    }
    return 0;
}
