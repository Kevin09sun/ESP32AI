#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_INA219.h>
#include "mlp_weights_i8.h"   // const (flash): MLP_C_q + scales, etc., ITOS

Adafruit_INA219 ina219;

// ---- protocol ----
const uint32_t CPU_MHZ   = 240;
const uint32_t SETTLE_MS = 5000;
const uint32_t SAMPLE_MS = 100;
const uint32_t N_SAMPLES = 100;
const float    CAL_K     = 0.801;  // measured 2026-08-04 vs multimeter
// ------------------

// ---- RAM copies of quantized weights ----
int8_t C_ram[27][10];
int8_t W1_ram[30][200];
int8_t b1_ram[200];
int8_t W2_ram[200][27];
int8_t b2_ram[27];

void copy_weights_to_ram() {
  memcpy(C_ram,  MLP_C_q,  sizeof(C_ram));
  memcpy(W1_ram, MLP_W1_q, sizeof(W1_ram));
  memcpy(b1_ram, MLP_b1_q, sizeof(b1_ram));
  memcpy(W2_ram, MLP_W2_q, sizeof(W2_ram));
  memcpy(b2_ram, MLP_b2_q, sizeof(b2_ram));
}

// ---- RNG ----
uint32_t rng_state = 42;
uint32_t xorshift32() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}
float rand01() { return (xorshift32() >> 8) * (1.0f / 16777216.0f); }

// ---- MLP state ----
int ctx[3] = {0, 0, 0};
float x[30], h[200], logits[27];
int ix = 0;

void build_input() {
  for (int t = 0; t < 3; t++)
    for (int e = 0; e < 10; e++)
      x[t * 10 + e] = C_ram[ctx[t]][e] * MLP_C_scale;
}

void forward() {
  build_input();

  for (int j = 0; j < 200; j++) {
    float acc = b1_ram[j] * MLP_b1_scale;
    for (int k = 0; k < 30; k++)
      acc += x[k] * (W1_ram[k][j] * MLP_W1_scale);
    h[j] = tanhf(acc);
  }

  for (int j = 0; j < 27; j++) {
    float acc = b2_ram[j] * MLP_b2_scale;
    for (int k = 0; k < 200; k++)
      acc += h[k] * (W2_ram[k][j] * MLP_W2_scale);
    logits[j] = acc;
  }
}

int step() {
  forward();

  float m = logits[0];
  for (int j = 1; j < 27; j++) if (logits[j] > m) m = logits[j];

  float p[27], sum = 0;
  for (int j = 0; j < 27; j++) { p[j] = expf(logits[j] - m); sum += p[j]; }

  float r = rand01() * sum;
  float c = 0;
  ix = 26;
  for (int j = 0; j < 27; j++) {
    c += p[j];
    if (r < c) { ix = j; break; }
  }
  ctx[0] = ctx[1]; ctx[1] = ctx[2]; ctx[2] = ix;
  if (ix == 0) ctx[0] = ctx[1] = ctx[2] = 0;
  return ix;
}

// ---- measurement state ----
double sumP = 0, sumP2 = 0, sumI = 0, sumV = 0;
uint32_t n = 0;
uint32_t chars = 0;
uint32_t elapsed_ms = 0;
bool done = false;

const int WITNESS_LEN = 200;
char names[WITNESS_LEN + 1];

void setup() {
  setCpuFrequencyMhz(CPU_MHZ);
  WiFi.mode(WIFI_OFF);
  btStop();

  Wire.begin(21, 22);
  ina219.begin();

  pinMode(2, OUTPUT);
  copy_weights_to_ram();
  delay(SETTLE_MS);

  digitalWrite(2, HIGH);
  uint32_t t_start   = millis();
  uint32_t next_samp = t_start;

  while (n < N_SAMPLES) {
    if (millis() >= next_samp) {
      float mV_bus = ina219.getBusVoltage_V() * 1000.0;
      float mA     = ina219.getCurrent_mA() * CAL_K;
      float mW     = (mV_bus / 1000.0) * mA;
      sumP += mW; sumP2 += (double)mW * mW;
      sumI += mA; sumV += mV_bus;
      n++;
      next_samp += SAMPLE_MS;
    }
    step();
    chars++;
  }
  elapsed_ms = millis() - t_start;

  ctx[0] = ctx[1] = ctx[2] = 0;
  for (int i = 0; i < WITNESS_LEN; i++)
    names[i] = ITOS[step()];
  names[WITNESS_LEN] = '\0';
  digitalWrite(2, LOW);

  done = true;
  Serial.begin(115200);
}

void loop() {
  if (!done) return;
  double meanP = sumP / n;
  double stdP  = sqrt(sumP2 / n - meanP * meanP);
  Serial.printf("MLP i8-RAM  CPU: %u MHz  n=%u  window=%u ms\n", CPU_MHZ, n, elapsed_ms);
  Serial.printf("Mean current: %.2f mA\n", sumI / n);
  Serial.printf("Mean bus V:   %.3f V\n", sumV / n / 1000.0);
  Serial.printf("Power: %.2f mW  +/- %.2f mW (std)\n", meanP, stdP);
  Serial.printf("Chars: %u   chars/sec: %.0f\n", chars, chars * 1000.0 / elapsed_ms);
  Serial.printf("J/char = (%.2f - baseline_mW) * %.3f s / %u / 1000\n",
                meanP, elapsed_ms / 1000.0, chars);
  Serial.printf("Witness names: %s\n", names);
  delay(4000);
}