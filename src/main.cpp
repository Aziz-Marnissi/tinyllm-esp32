#include <Arduino.h>
#include <ESP32Servo.h>
#include <DHT.h>
#include "model_api.h"

// ---- pins ----
#define LED_PIN     2
#define SERVO_PIN   5
#define DHT_PIN     4
#define DHT_TYPE    DHT11
#define MOTOR_IN1   16
#define MOTOR_IN2   17
#define MOTOR_IN3   18
#define MOTOR_IN4   19

const char* ACTION_NAMES[N_ACTIONS] = {"on", "off", "set_power", "set_degree", "set_timer", "check_status"};
const char* TARGET_NAMES[N_TARGETS] = {"led", "motor_28BYJ48", "servo", "timer", "temp_sensor"};

Servo myServo;
DHT dht(DHT_PIN, DHT_TYPE);

// ---- 28BYJ48 half-step sequence via ULN2003 ----
const uint8_t STEP_SEQ[8][4] = {
    {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
    {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1}
};
int stepIndex = 0;
bool motorRunning = false;

void motorStepOnce() {
    digitalWrite(MOTOR_IN1, STEP_SEQ[stepIndex][0]);
    digitalWrite(MOTOR_IN2, STEP_SEQ[stepIndex][1]);
    digitalWrite(MOTOR_IN3, STEP_SEQ[stepIndex][2]);
    digitalWrite(MOTOR_IN4, STEP_SEQ[stepIndex][3]);
    stepIndex = (stepIndex + 1) % 8;
}

void motorOff() {
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
    digitalWrite(MOTOR_IN3, LOW);
    digitalWrite(MOTOR_IN4, LOW);
    motorRunning = false;
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    pinMode(MOTOR_IN1, OUTPUT);
    pinMode(MOTOR_IN2, OUTPUT);
    pinMode(MOTOR_IN3, OUTPUT);
    pinMode(MOTOR_IN4, OUTPUT);
    myServo.attach(SERVO_PIN);
    dht.begin();
    Serial.println("ready");
}

void loop() {
    // keep stepping while motor is "on" -- non-blocking
    if (motorRunning) {
        motorStepOnce();
        delay(2); // ~ safe step rate for 28BYJ48
    }

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
    float value_scaled = value_out * 180.0f; // MAX_VALUE du modele Python

    Serial.printf("action=%s target=%s value=%.1f  (%.2f ms)\n",
                  ACTION_NAMES[best_a], TARGET_NAMES[best_t], value_scaled, dt_us / 1000.0f);

    // ---- routage vers le materiel ----
    if (best_t == 0) { // led
        if (best_a == 0) digitalWrite(LED_PIN, HIGH);
        else if (best_a == 1) digitalWrite(LED_PIN, LOW);
        // set_power: necessite PWM (ledcWrite) -- a ajouter si besoin
    } else if (best_t == 1) { // motor_28BYJ48
        if (best_a == 0) motorRunning = true;
        else if (best_a == 1) motorOff();
    } else if (best_t == 2) { // servo
        if (best_a == 3) { // set_degree
            int deg = constrain((int)value_scaled, 0, 180);
            myServo.write(deg);
        }
    } else if (best_t == 3) { // timer
        // set_timer: demarrer un compte a rebours logiciel -- a implementer si besoin
    } else if (best_t == 4) { // temp_sensor
        if (best_a == 5) { // check_status
            float t = dht.readTemperature();
            if (isnan(t)) {
                Serial.println("DHT11 read failed");
            } else {
                Serial.printf("temperature=%.1f C\n", t);
            }
        }
    }
}
