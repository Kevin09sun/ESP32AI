#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_INA219.h>

// ---- pick model + storage per flash ----
//#include "wavenet_e24_h128_f32.h"   // arm (b): 76k, flash-only
#include "wavenet_e16_h80_f32.h" // arm (a): 31k, use RAM
#define USE_RAM 1                    // 0 = flash-resident (arm b), 1 = RAM copies (arm a)
// -----------------------------------------

Adafruit_INA219 ina219;

const uint32_t CPU_MHZ   = 240;
const uint32_t SETTLE_MS = 5000;
const uint32_t SAMPLE_MS = 100;
const uint32_t N_SAMPLES = 100;
const float    CAL_K     = 0.801;  // measured 2026-08-04 vs multimeter

// dims from the header
#define E  WN_EMBD
#define H  WN_HIDDEN

#if USE_RAM
float C_ram[27][E];
float (*W1_ram)[H];      // (2E x H)
float b1_ram[H];
float (*W2_ram)[H];      // (2H x H)
float b2_ram[H];
float (*W3_ram)[H];      // (2H x H)
float b3_ram[H];
float Wout_ram[H][27];   // small enough static
float bout_ram[27];

bool copy_to_ram() {
  W1_ram = (float(*)[H]) malloc(sizeof(float) * 2 * E * H);
  W2_ram = (float(*)[H]) malloc(sizeof(float) * 2 * H * H);
  W3_ram = (float(*)[H]) malloc(sizeof(float) * 2 * H * H);
  if (!W1_ram || !W2_ram || !W3_ram) return false;
  memcpy(C_ram,  WN_C,  sizeof(C_ram));
  memcpy(W1_ram, WN_W1, sizeof(float) * 2 * E * H);
  memcpy(b1_ram, WN_b1, sizeof(b1_ram));
  memcpy(W2_ram, WN_W2, sizeof(float) * 2 * H * H);
  memcpy(b2_ram, WN_b2, sizeof(b2_ram));
  memcpy(W3_ram, WN_W3, sizeof(float) * 2 * H * H);
  memcpy(b3_ram, WN_b3, sizeof(b3_ram));
  memcpy(Wout_ram, WN_Wout, sizeof(Wout_ram));
  memcpy(bout_ram, WN_bout, sizeof(bout_ram));
  return true;
}
#define M_C(i,j)    C_ram[i][j]
#define M_W1(i,j)   W1_ram[i][j]
#define M_B1(j)     b1_ram[j]
#define M_W2(i,j)   W2_ram[i][j]
#define M_B2(j)     b2_ram[j]
#define M_W3(i,j)   W3_ram[i][j]
#define M_B3(j)     b3_ram[j]
#define M_WO(i,j)   Wout_ram[i][j]
#define M_BO(j)     bout_ram[j]
#else
#define M_C(i,j)    WN_C[i][j]
#define M_W1(i,j)   WN_W1[i][j]
#define M_B1(j)     WN_b1[j]
#define M_W2(i,j)   WN_W2[i][j]
#define M_B2(j)     WN_b2[j]
#define M_W3(i,j)   WN_W3[i][j]
#define M_B3(j)     WN_b3[j]
#define M_WO(i,j)   WN_Wout[i][j]
#define M_BO(j)     WN_bout[j]
#endif

// ---- RNG ----
uint32_t rng_state = 42;
uint32_t xorshift32() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}
float rand01() { return (xorshift32() >> 8) * (1.0f / 16777216.0f); }

// ---- wavenet inference ----
// context: 8 previous char indices (oldest first), like Python's context list
int ctx[8] = {0,0,0,0,0,0,0,0};
int ix = 0;

float emb[8][E];      // embedded context
float h4[4][H];       // after level 1 (4 positions)
float h2[2][H];       // after level 2 (2 positions)
float h1[H];          // after level 3 (1 position)
float logits[27];

void forward() {
  // embed
  for (int t = 0; t < 8; t++)
    for (int e = 0; e < E; e++)
      emb[t][e] = M_C(ctx[t], e);

  // level 1: pairs (0,1),(2,3),(4,5),(6,7) -> concat 2E -> H, tanh
  for (int p = 0; p < 4; p++) {
    for (int j = 0; j < H; j++) {
      float acc = M_B1(j);
      for (int k = 0; k < E; k++) {
        acc += emb[2*p][k]   * M_W1(k, j);        // first of pair -> rows 0..E-1
        acc += emb[2*p+1][k] * M_W1(E + k, j);    // second of pair -> rows E..2E-1
      }
      h4[p][j] = tanhf(acc);
    }
  }

  // level 2: pairs of h4 -> concat 2H -> H, tanh
  for (int p = 0; p < 2; p++) {
    for (int j = 0; j < H; j++) {
      float acc = M_B2(j);
      for (int k = 0; k < H; k++) {
        acc += h4[2*p][k]   * M_W2(k, j);
        acc += h4[2*p+1][k] * M_W2(H + k, j);
      }
      h2[p][j] = tanhf(acc);
    }
  }

  // level 3: the last pair -> H, tanh
  for (int j = 0; j < H; j++) {
    float acc = M_B3(j);
    for (int k = 0; k < H; k++) {
      acc += h2[0][k] * M_W3(k, j);
      acc += h2[1][k] * M_W3(H + k, j);
    }
    h1[j] = tanhf(acc);
  }

  // output
  for (int j = 0; j < 27; j++) {
    float acc = M_BO(j);
    for (int k = 0; k < H; k++)
      acc += h1[k] * M_WO(k, j);
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
  for (int t = 0; t < 7; t++) ctx[t] = ctx[t+1];   // shift window
  ctx[7] = ix;
  if (ix == 0) for (int t = 0; t < 8; t++) ctx[t] = 0;   // word boundary reset
  return ix;
}

// ---- measurement (unchanged scaffold) ----
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

#if USE_RAM
  if (!copy_to_ram()) {
    while (true) {   // SOS: allocation failed
      for (int i = 0; i < 3; i++) { digitalWrite(2, HIGH); delay(150); digitalWrite(2, LOW); delay(150); }
      delay(700);
    }
  }
#endif

  delay(SETTLE_MS);
  digitalWrite(2, HIGH);
  uint32_t t_start = millis();
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

  for (int t = 0; t < 8; t++) ctx[t] = 0;
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
  Serial.printf("WAVENET e=%d h=%d %s  CPU: %u MHz  n=%u  window=%u ms\n",
                E, H, USE_RAM ? "RAM" : "FLASH", CPU_MHZ, n, elapsed_ms);
  Serial.printf("Mean current: %.2f mA\n", sumI / n);
  Serial.printf("Mean bus V:   %.3f V\n", sumV / n / 1000.0);
  Serial.printf("Power: %.2f mW  +/- %.2f mW (std)\n", meanP, stdP);
  Serial.printf("Chars: %u   chars/sec: %.0f\n", chars, chars * 1000.0 / elapsed_ms);
  Serial.printf("J/char = (%.2f - baseline_mW) * %.3f s / %u / 1000\n",
                meanP, elapsed_ms / 1000.0, chars);
  Serial.printf("Witness names: %s\n", names);
  delay(4000);
}