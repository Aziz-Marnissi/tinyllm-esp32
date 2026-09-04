#include <Arduino.h>
#include "model_api.h"
// ---- pins (à adapter à ton câblage) ----
#define LED_PIN     2
#define MOTOR_PIN   4   // placeholder simple on/off; le 28BYJ48 réel nécessite un séquenceur 4 phases
#define SERVO_PIN   5
const char* ACTION_NAMES[N_ACTIONS] = {"on", "off", "set_power", "set_degree", "set_timer", "check_status"};
const char* TARGET_NAMES[N_TARGETS] = {"led", "motor_28BYJ48", "servo", "timer", "temp_sensor"};
void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    pinMode(MOTOR_PIN, OUTPUT);
    pinMode(SERVO_PIN, OUTPUT); // remplacer par ESP32Servo pour un vrai PWM servo
    Serial.println("ready");
}
void loop() {
    if (!Serial.available()) return;
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;
    int ids[MAX_LEN];
    int length = tokenize(line.c_str(), ids);
    float num_raw, num_feat;
    int num_present;
    extract_number(line.c_str(), &num_raw, &num_present);
    num_feat = num_raw / 180.0f;
    float action_logits[N_ACTIONS], target_logits[N_TARGETS], value_out;
    unsigned long t0 = micros();
    model_forward(ids, length, num_feat, (float)num_present, action_logits, target_logits, &value_out);
    unsigned long dt_us = micros() - t0;
    int best_a = 0, best_t = 0;
    for (int i = 1; i < N_ACTIONS; i++) if (action_logits[i] > action_logits[best_a]) best_a = i;
    for (int i = 1; i < N_TARGETS; i++) if (target_logits[i] > target_logits[best_t]) best_t = i;
    float value_scaled = value_out * 180.0f; // MAX_VALUE du modèle Python
    Serial.printf("action=%s target=%s value=%.1f  (%.2f ms)\n",
                  ACTION_NAMES[best_a], TARGET_NAMES[best_t], value_scaled, dt_us / 1000.0f);
    // ---- routage vers le matériel ----
    if (best_t == 0) { // led
        if (best_a == 0) digitalWrite(LED_PIN, HIGH);
        else if (best_a == 1) digitalWrite(LED_PIN, LOW);
        // set_power: nécessite PWM (ledcWrite) -- à ajouter si besoin
    } else if (best_t == 1) { // motor_28BYJ48
        if (best_a == 0) digitalWrite(MOTOR_PIN, HIGH);
        else if (best_a == 1) digitalWrite(MOTOR_PIN, LOW);
    } else if (best_t == 2) { // servo
        // set_degree: nécessite ESP32Servo.h -- à brancher ici
    } else if (best_t == 3) { // timer
        // set_timer: démarrer un compte à rebours logiciel -- à implémenter si besoin
    } else if (best_t == 4) { // temp_sensor
        // check_status: lire un capteur réel (DHT22 etc.) -- à brancher ici
    }
}
