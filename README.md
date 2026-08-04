# How much energy does a language model cost, per character?

Measuring the energy cost of running small language models on a microcontroller — from a bigram lookup table to a trained MLP — to find the cheapest hardware configuration that produces "usable" text generation.

## The question

Big language models run in datacenters where energy is measured in megawatts. What does inference cost at the *bottom* of the hardware ladder — a $10 microcontroller — and where does capability-per-joule stop being worth it? I measure joules per character across model types, quantization schemes, clock speeds, and model sizes on an ESP32, and define concretely what "usable" means.

## The rig

- ESP32 dev board (Elegoo kit), powered through an INA219 current sensor on the 5 V supply line
- The ESP32 reads its own supply current over I2C (documented limitation — see Error analysis)
- Breadboard power supply module → INA219 shunt → ESP32 VIN; all measurement runs with USB disconnected (single power source)
- Multimeter used as independent reference instrument

<!-- TODO: wiring photo + diagram (Phase 8) -->

## Method

Power is sampled at 10 Hz for 10 s after a 5 s settle, WiFi/BT explicitly disabled. Energy per character = (mean power under load − idle baseline) ÷ characters generated. All currents corrected by a measured calibration factor (see Instrument validation).

<!-- One sentence justifying each protocol choice: settle time, sample rate, window length, idle definition. Yours. -->

## 1. Baseline: idle power vs CPU frequency

| CPU clock | Mean current | Bus voltage | Power | Std |
|---|---|---|---|---|
| 240 MHz | 56.36 mA | 4.982 V | 280.81 mW | ±5.23 mW |
| 160 MHz | 49.82 mA | 4.986 V | 248.39 mW | ±4.01 mW |
| 80 MHz  | 42.69 mA | 4.994 V | 213.17 mW | ±4.25 mW |

n = 100 samples per run, 10 Hz, corrected current (k = 0.801, see below).

Observations:
- Power decreases monotonically and near-linearly with clock: ~0.53 mW/MHz
- Naive linear extrapolation to 0 MHz suggests a fixed overhead floor of roughly 178 mW (LDO, power LED, peripherals, leakage) — ~63% of total power at 240 MHz. This is why all inference measurements are reported as deltas above baseline.
- Repeatability: runs taken on different occasions agree within ~1%.

## Instrument validation

Predicted current through a measured 220.0 Ω resistor at the measured rail voltage: 22.7–22.9 mA. Multimeter in series read 22.2 mA. The INA219 reported 27.70 mA — a +25% over-read.

Diagnosis: shunt confirmed R100 by marking; switching library calibration range (32V/2A → 16V/400mA) had no effect. Root cause undiagnosed (plausibly shunt tolerance or clone INA219 die); correction is empirical.

Fix: measured scale factor k = 22.2 / 27.70 = 0.801 applied to all current readings in software. Verified at a second load point (330 Ω: predicted 15.2 mA, corrected reading 15.0 mA, ~1.5% agreement) — gain correction is linear across 15–22 mA.

<!-- Yours: write this as a paragraph in your own words. The % disagreement number is the checklist requirement. -->

## Error analysis

- **Two-source contamination:** with USB connected, current reads negative (reverse flow through the shunt). All measurements are USB-disconnected; positive current is a per-run validity check.
- **Calibration extrapolation:** calibration points span 15–22 mA; operating currents are 42–56 mA. Correction is assumed linear beyond the calibrated range. (Optional closure: a ~50 mA calibration point via 100 Ω load.)
- **Bus voltage vs rail:** INA219 bus voltage reads ~60 mV below the multimeter rail measurement — shunt drop plus breadboard contact resistance. The ESP32 sees slightly under 5 V.
- **Noise floor:** idle power std is ±4–6 mW (~2%). Inference deltas near this magnitude are at the edge of resolution.
- **Window length:** 10 s windows won't capture slow drifts (e.g. regulator warm-up). Spot-check longer runs if results look suspicious.
- **Self-power:** the INA219 is powered from the ESP32's 3.3 V rail, so its own ~1 mA draw flows through the shunt and is counted in every measurement, including baseline — it cancels in the delta.
- **Root cause of the 25% gain error remains unknown** — the correction is calibration, not explanation.

<!-- Expand each bullet to a sentence or two in your voice. -->

## 2. Bigram model (embedding/lookup floor)

<!-- Phase 1 results land here the day they exist. -->

## 3. Matmul microbenchmark

<!-- Phase 2: J per million MACs, int8 vs float32. -->

## 4. MLP (makemore)

<!-- Phase 3: J/char, chars/sec, cross-check against Phase 2 prediction. -->

## 5. Sweeps

<!-- Phase 4: quantization / clock / model size. One chart + one sentence each. -->

## 6. What counts as "usable," and what it costs

<!-- Phase 6: the definition, the frontier chart, the verdict paragraph. -->

## Incident log

- **Negative current with USB attached** — symptom: −19 mA at idle; measurement: sign flip on USB disconnect; cause: two active power sources, USB backfeeding through the shunt path; fix: USB-disconnected protocol, positive-current validity check.
- **INA219 +25% over-read** — symptom: 27.7 mA on a 22.9 mA predicted load; measurement: multimeter series reference at two load points; cause: undiagnosed gain error (shunt marking correct, calibration range irrelevant); fix: empirical scale factor 0.801, linear-verified.

<!-- Add every future incident the day it happens. Brownouts go here. -->

## What surprised me

<!-- Write honestly, last. Candidates so far: the instrument was 25% wrong out of the box; idle overhead dwarfs the CPU's own draw. -->

## Reproducibility

<!-- Phase 8: wiring photo, how-to-run, code map. -->