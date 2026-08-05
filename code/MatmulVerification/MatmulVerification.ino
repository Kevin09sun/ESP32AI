#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_INA219.h>

Adafruit_INA219 ina219;

// ---- protocol ----
const uint32_t CPU_MHZ   = 240;
const uint32_t SETTLE_MS = 5000;   // once, before the first config
const uint32_t GAP_MS    = 2000;   // between configs
const uint32_t SAMPLE_MS = 100;    // 10 Hz
const uint32_t N_SAMPLES = 100;    // 10 s window per config
const float    CAL_K     = 0.801;  // measured 2026-08-04 vs multimeter
// ------------------

const int MAXN = 64;

// flat buffers, indexed [i*N + j] with runtime N
float   Af[MAXN * MAXN], Bf[MAXN * MAXN], Cf[MAXN * MAXN];
int8_t  Ai[MAXN * MAXN], Bi[MAXN * MAXN];
int32_t Ci[MAXN * MAXN];

uint32_t rng_state;

uint32_t xorshift32() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

void fill_matrices(int N) {
  rng_state = 67;   // same seed every config -> N=16 inputs are a subset-pattern, reproducible
  for (int i = 0; i < N * N; i++) {
    uint32_t r = xorshift32();
    Ai[i] = (int8_t)(r & 0xFF);
    Af[i] = (int8_t)(r & 0xFF) / 128.0f;
  }
  for (int i = 0; i < N * N; i++) {
    uint32_t r = xorshift32();
    Bi[i] = (int8_t)(r & 0xFF);
    Bf[i] = (int8_t)(r & 0xFF) / 128.0f;
  }
}

void matmul_f32(int N) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      float acc = 0;
      for (int k = 0; k < N; k++)
        acc += Af[i * N + k] * Bf[k * N + j];
      Cf[i * N + j] = acc;
    }
}

void matmul_int8(int N) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      int32_t acc = 0;
      for (int k = 0; k < N; k++)
        acc += Ai[i * N + k] * Bi[k * N + j];
      Ci[i * N + j] = acc;
    }
}

volatile uint32_t g_check_i = 0;
volatile float    g_check_f = 0;

void run_once(bool use_int8, int N) {
  if (use_int8) {
    matmul_int8(N);
    uint32_t x = 0;
    for (int i = 0; i < N * N; i++) x ^= (uint32_t)Ci[i];
    g_check_i += x;                 // += so identical reps don't cancel
  } else {
    matmul_f32(N);
    float s = 0;
    for (int i = 0; i < N * N; i++) s += Cf[i];
    g_check_f += s;
  }
}

// ---- per-config results ----
struct Result {
  bool     use_int8;
  int      N;
  double   meanI, meanV, meanP, stdP;
  uint32_t reps, window_ms, samples;
  uint32_t check_i;
  float    check_f;
};

const int N_CONFIGS = 6;
Result results[N_CONFIGS];
bool done = false;

void run_config(int idx, bool use_int8, int N) {
  fill_matrices(N);
  g_check_i = 0;
  g_check_f = 0;

  double sumP = 0, sumP2 = 0, sumI = 0, sumV = 0;
  uint32_t n = 0, reps = 0;

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
    run_once(use_int8, N);
    reps++;
  }
  uint32_t elapsed = millis() - t_start;
  digitalWrite(2, LOW);

  Result& r = results[idx];
  r.use_int8 = use_int8; r.N = N;
  r.meanI = sumI / n; r.meanV = sumV / n / 1000.0;
  r.meanP = sumP / n;
  r.stdP  = sqrt(sumP2 / n - r.meanP * r.meanP);
  r.reps  = reps; r.window_ms = elapsed; r.samples = n;
  r.check_i = g_check_i; r.check_f = g_check_f;
}

void setup() {
  setCpuFrequencyMhz(CPU_MHZ);
  WiFi.mode(WIFI_OFF);
  btStop();

  Wire.begin(21, 22);
  ina219.begin();
  pinMode(2, OUTPUT);

  delay(SETTLE_MS);

  const int  sizes[3] = {16, 32, 64};
  int idx = 0;
  for (int t = 0; t < 2; t++) {            // t=0: int8, t=1: f32
    for (int s = 0; s < 3; s++) {
      run_config(idx++, t == 0, sizes[s]);
      delay(GAP_MS);
    }
  }
  done = true;
  Serial.begin(115200);
}

void loop() {
  if (!done) return;
  Serial.printf("=== MATMUL SWEEP  CPU %u MHz  window %u samples @ %u ms ===\n",
                CPU_MHZ, N_SAMPLES, SAMPLE_MS);
  for (int i = 0; i < N_CONFIGS; i++) {
    Result& r = results[i];
    double macs  = (double)r.reps * r.N * r.N * r.N;
    double secs  = r.window_ms / 1000.0;
    Serial.printf("[%s N=%2d] P=%.2f +/- %.2f mW  I=%.2f mA  V=%.3f V  "
                  "reps=%lu  MMACs/s=%.2f  win=%lu ms  check=%08X/%.3f\n",
                  r.use_int8 ? "int8" : "f32 ", r.N,
                  r.meanP, r.stdP, r.meanI, r.meanV,
                  r.reps, macs / 1e6 / secs, r.window_ms,
                  r.check_i, r.check_f);
  }
  Serial.println("J/MMAC = (P - baseline) * win_s / MMACs / 1000");
  delay(5000);
}