#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_INA219.h>
#include "bigram_weights.h"

Adafruit_INA219 ina219;

// ---- protocol ----
const uint32_t CPU_MHZ     = 240;   // change per run: 240 / 160 / 80
const uint32_t SETTLE_MS   = 5000;
const uint32_t SAMPLE_MS   = 100;   // 10 Hz
const uint32_t N_SAMPLES   = 100;   // 60 s
const float CAL_K = 0.801;
// ------------------

// RNG so we have reproducible runs
uint32_t rng_state = 42;

uint32_t xorshift32() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

float rand01() {
  return (xorshift32() >> 8) * (1.0f / 16777216.0f);  // 24 mantissa-width bits -> [0,1)
}

int ix = 0;
int step() {
  const float* logits = BIGRAM_W[ix];   // one-hot @ W == row selection

  float m = logits[0];
  for (int j = 1; j < 27; j++) if (logits[j] > m) m = logits[j];

  float p[27], sum = 0;
  for (int j = 0; j < 27; j++) { p[j] = expf(logits[j] - m); sum += p[j]; }

  float r = rand01() * sum;             // sample against unnormalized cumsum
  float c = 0;
  for (int j = 0; j < 27; j++) {
    c += p[j];
    if (r < c) { ix = j; return j; }
  }
  ix = 26; return 26;                   // r == sum float edge case
}

double sumP = 0, sumP2 = 0, sumI = 0, sumV = 0;
uint32_t n = 0;
uint32_t chars = 0;
uint32_t elapsed_ms = 0;
bool done = false;

void setup() {
  setCpuFrequencyMhz(CPU_MHZ);
  WiFi.mode(WIFI_OFF);
  btStop();

  Wire.begin(21, 22);
  ina219.begin();

  pinMode(2, OUTPUT);          // onboard LED on most ESP32 dev boards
  delay(SETTLE_MS);

  digitalWrite(2, HIGH);        // LED on = measuring
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
      next_samp += SAMPLE_MS;           // fixed cadence, no drift
    }
    step();
    chars++;  
  }
  elapsed_ms = millis() - t_start;
  digitalWrite(2, LOW);                 // LED off = done
  done = true;

  Serial.begin(115200);
}

void loop() {
  if (!done) return;
  double meanP = sumP / n;
  double stdP  = sqrt(sumP2 / n - meanP * meanP);
  Serial.printf("BIGRAM  CPU: %u MHz  n=%u  window=%u ms\n", CPU_MHZ, n, elapsed_ms);
  Serial.printf("Mean current: %.2f mA\n", sumI / n);
  Serial.printf("Mean bus V:   %.3f V\n", sumV / n / 1000.0);
  Serial.printf("Power: %.2f mW  +/- %.2f mW (std)\n", meanP, stdP);
  Serial.printf("Chars: %u   chars/sec: %.0f\n", chars, chars * 1000.0 / elapsed_ms);
  Serial.printf("J/char = (%.2f - baseline_mW) * %.3f s / %u / 1000\n", meanP, elapsed_ms / 1000.0, chars);
  delay(4000);
}