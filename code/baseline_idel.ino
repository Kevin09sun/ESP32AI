#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_INA219.h>

Adafruit_INA219 ina219;

// ---- protocol ----
const uint32_t CPU_MHZ     = 80;   // change per run: 240 / 160 / 80
const uint32_t SETTLE_MS   = 5000;
const uint32_t SAMPLE_MS   = 100;   // 10 Hz
const uint32_t N_SAMPLES   = 100;   // 60 s
const float CAL_K = 0.801;
// ------------------

double sumP = 0, sumP2 = 0, sumI = 0, sumV = 0;
double minI = 1e9, maxI = -1e9;
uint32_t n = 0;
bool done = false;

void setup() {
  setCpuFrequencyMhz(CPU_MHZ);
  WiFi.mode(WIFI_OFF);
  btStop();

  Wire.begin(21, 22);
  ina219.begin();

  pinMode(2, OUTPUT);          // onboard LED on most ESP32 dev boards
  delay(SETTLE_MS);

  digitalWrite(2, HIGH);       // LED on = measuring
  while (n < N_SAMPLES) {
    float mV_bus = ina219.getBusVoltage_V() * 1000.0;
    float mA     = ina219.getCurrent_mA() * CAL_K;
    float mW     = (mV_bus / 1000.0) * mA;

    sumI += mA;  sumV += mV_bus;
    sumP += mW;  sumP2 += (double)mW * mW;
    if (mA < minI) minI = mA;
    if (mA > maxI) maxI = mA;
    n++;
    delay(SAMPLE_MS);
  }
  digitalWrite(2, LOW);        // LED off = done, safe to replug
  done = true;

  Serial.begin(115200);
}

void loop() {
  if (!done) return;
  double meanP = sumP / n;
  double stdP  = sqrt(sumP2 / n - meanP * meanP);
  Serial.printf("CPU: %u MHz  n=%u\n", CPU_MHZ, n);
  Serial.printf("Mean current: %.2f mA (min %.2f / max %.2f)\n", sumI / n, minI, maxI);
  Serial.printf("Mean bus V:   %.3f V\n", sumV / n / 1000.0);
  Serial.printf("Power: %.2f mW  +/- %.2f mW (std)\n", meanP, stdP);
  delay(2000);
}