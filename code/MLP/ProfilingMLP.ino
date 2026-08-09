#include <WiFi.h>
#include "mlp_weights_i8.h"

// ---- profiling sketch, int8-dequant edition: where did the 2x speedup come from? ----
// USB stays connected; no INA219.

const uint32_t CPU_MHZ = 240;
const int RUNS = 1000;

uint32_t rng_state = 42;
uint32_t xorshift32() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}
float rand01() { return (xorshift32() >> 8) * (1.0f / 16777216.0f); }

int ctx[3] = {0, 0, 0};
float x[30], h[200], logits[27];
int ix = 0;

void build_input() {
  for (int t = 0; t < 3; t++)
    for (int e = 0; e < 10; e++)
      x[t * 10 + e] = MLP_C_q[ctx[t]][e] * MLP_C_scale;
}

void setup() {
  setCpuFrequencyMhz(CPU_MHZ);
  WiFi.mode(WIFI_OFF);
  btStop();
  Serial.begin(115200);
  delay(2000);

  uint64_t t_w1 = 0, t_tanh = 0, t_w2 = 0, t_sm = 0;
  volatile float sink = 0;

  for (int r = 0; r < RUNS; r++) {
    build_input();

    uint32_t t0 = micros();

    // --- W1 matmul + bias, dequant-on-load (tanh split out) ---
    for (int j = 0; j < 200; j++) {
      float acc = MLP_b1_q[j] * MLP_b1_scale;
      for (int k = 0; k < 30; k++)
        acc += x[k] * (MLP_W1_q[k][j] * MLP_W1_scale);
      h[j] = acc;
    }
    uint32_t t1 = micros();

    // --- tanh alone, 200 calls ---
    for (int j = 0; j < 200; j++)
      h[j] = tanhf(h[j]);
    uint32_t t2 = micros();

    // --- W2 matmul + bias, dequant-on-load ---
    for (int j = 0; j < 27; j++) {
      float acc = MLP_b2_q[j] * MLP_b2_scale;
      for (int k = 0; k < 200; k++)
        acc += h[k] * (MLP_W2_q[k][j] * MLP_W2_scale);
      logits[j] = acc;
    }
    uint32_t t3 = micros();

    // --- softmax + sample ---
    float m = logits[0];
    for (int j = 1; j < 27; j++) if (logits[j] > m) m = logits[j];
    float p[27], sum = 0;
    for (int j = 0; j < 27; j++) { p[j] = expf(logits[j] - m); sum += p[j]; }
    float rr = rand01() * sum;
    float c = 0;
    ix = 26;
    for (int j = 0; j < 27; j++) { c += p[j]; if (rr < c) { ix = j; break; } }
    uint32_t t4 = micros();

    ctx[0] = ctx[1]; ctx[1] = ctx[2]; ctx[2] = ix;
    if (ix == 0) ctx[0] = ctx[1] = ctx[2] = 0;

    t_w1   += t1 - t0;
    t_tanh += t2 - t1;
    t_w2   += t3 - t2;
    t_sm   += t4 - t3;
    sink += h[0] + logits[0];
  }

  float fw1 = t_w1 / (float)RUNS, ftanh = t_tanh / (float)RUNS;
  float fw2 = t_w2 / (float)RUNS, fsm = t_sm / (float)RUNS;
  float tot = fw1 + ftanh + fw2 + fsm;

  Serial.printf("=== MLP i8-deq profile, %d runs @ %u MHz ===\n", RUNS, CPU_MHZ);
  Serial.printf("W1 matmul (6000 MACs): %7.1f us  (%4.1f%%)   [f32 was 1002.3]\n", fw1,   100 * fw1 / tot);
  Serial.printf("tanh x200:             %7.1f us  (%4.1f%%)   [f32 was  405.7]\n", ftanh, 100 * ftanh / tot);
  Serial.printf("W2 matmul (5400 MACs): %7.1f us  (%4.1f%%)   [f32 was  853.8]\n", fw2,   100 * fw2 / tot);
  Serial.printf("softmax+sample:        %7.1f us  (%4.1f%%)   [f32 was   82.3]\n", fsm,   100 * fsm / tot);
  Serial.printf("total: %.1f us/char -> implies %.0f chars/sec  [f32: 2344.0, 427]\n", tot, 1e6f / tot);
  Serial.printf("(measured i8 sketch: 1157 us/char, 864 chars/sec)\n");
}

void loop() { delay(5000); }