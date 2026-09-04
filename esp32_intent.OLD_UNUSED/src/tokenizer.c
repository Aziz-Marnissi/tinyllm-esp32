#include <ctype.h>
#include <string.h>
#include "vocab.h"
#define MAX_LEN 12
#define UNK_ID 1
#define PAD_ID 0

static int lookup_word(const char* word) {
    for (int i = 0; i < VOCAB_N; i++) {
        if (strcmp(VOCAB[i].word, word) == 0) return VOCAB[i].id;
    }
    return UNK_ID;
}
static int is_all_digits(const char* s, int len) {
    for (int i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return 0;
    }
    return 1;
}

// Tokenizes `text` and writes up to MAX_LEN ids into out_ids (padded with 0).
// Returns the REAL (unpadded) token count -- needed by the bidirectional
// GRU backward pass, which must start from the last real token, not the
// last padding slot.
int tokenize(const char* text, int out_ids[MAX_LEN]) {
    for (int i = 0; i < MAX_LEN; i++) out_ids[i] = PAD_ID;
    char buf[32];
    int buf_len = 0;
    int n_tokens = 0;
    int len = strlen(text);
    for (int i = 0; i <= len && n_tokens < MAX_LEN; i++) {
        char c = text[i];
        char lc = (char)tolower((unsigned char)c);
        int is_alnum = (i < len) && ((lc >= 'a' && lc <= 'z') || (lc >= '0' && lc <= '9'));
        if (is_alnum && buf_len < (int)sizeof(buf) - 1) {
            buf[buf_len++] = lc;
        } else if (buf_len > 0) {
            buf[buf_len] = '\0';
            if (is_all_digits(buf, buf_len)) {
                out_ids[n_tokens++] = lookup_word("<num>");
            } else {
                out_ids[n_tokens++] = lookup_word(buf);
            }
            buf_len = 0;
        }
    }
    return n_tokens;
}
