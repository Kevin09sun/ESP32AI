// Transformer inference core. Pure C, no Arduino dependencies, so the identical
// code can be compiled on the host and checked against the numpy reference.
#pragma once
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "transformer_weights.h"

#if TFM_LAYERS != 2
#error "bind macros below are written for exactly 2 layers; extend BIND_FLASH/BIND_RAM"
#endif

#define E    TFM_EMBD
#define FF   (4 * TFM_EMBD)
#define NL   TFM_LAYERS
#define NH   TFM_HEADS
#define HS   TFM_HEADSZ
#define BS   TFM_BLOCK
#define VOC  TFM_VOCAB

typedef struct {
  const float *ln1_g, *ln1_b, *ln2_g, *ln2_b, *o_b, *fc_b, *pr_b;
  const float (*q)[E];
  const float (*k)[E];
  const float (*v)[E];
  const float (*o)[E];
  const float (*fc)[FF];
  const float (*pr)[E];
} TfmLayer;

static TfmLayer LY[NL];
static const float (*TOKE)[E];
static const float (*POSE)[E];
static const float (*HEADW)[VOC];
static const float *HEAD_B, *LNF_G, *LNF_B;

// KV cache: 2 * NL * BS * E floats = 8 KB at the default config.
static float kc[NL][BS][E];
static float vc[NL][BS][E];

// ---- weight binding -------------------------------------------------------
// Flash: point straight at the const arrays (they live in flash behind the cache).
#define BIND_FLASH_L(L)                        \
  LY[L].ln1_g = TFM_l##L##_ln1_g;              \
  LY[L].ln1_b = TFM_l##L##_ln1_b;              \
  LY[L].ln2_g = TFM_l##L##_ln2_g;              \
  LY[L].ln2_b = TFM_l##L##_ln2_b;              \
  LY[L].o_b   = TFM_l##L##_o_b;                \
  LY[L].fc_b  = TFM_l##L##_fc_b;               \
  LY[L].pr_b  = TFM_l##L##_pr_b;               \
  LY[L].q     = TFM_l##L##_q;                  \
  LY[L].k     = TFM_l##L##_k;                  \
  LY[L].v     = TFM_l##L##_v;                  \
  LY[L].o     = TFM_l##L##_o;                  \
  LY[L].fc    = TFM_l##L##_fc;                 \
  LY[L].pr    = TFM_l##L##_pr;

static int tfm_bind_flash(void) {
  BIND_FLASH_L(0)
  BIND_FLASH_L(1)
  TOKE = TFM_tok_emb; POSE = TFM_pos_emb; HEADW = TFM_head;
  HEAD_B = TFM_head_b; LNF_G = TFM_lnf_g; LNF_B = TFM_lnf_b;
  return 1;
}

// RAM: small vectors into static arrays, six big matrices per layer onto the heap.
// Heap rather than static for the big ones because that is the route that worked
// for the wavenet at a comparable footprint (~112 KB).
static float r_tok[VOC][E], r_pos[BS][E], r_head[E][VOC];
static float r_head_b[VOC], r_lnf_g[E], r_lnf_b[E];
static float r_ln1g[NL][E], r_ln1b[NL][E], r_ln2g[NL][E], r_ln2b[NL][E];
static float r_ob[NL][E], r_fcb[NL][FF], r_prb[NL][E];

static void *tfm_dup(const void *src, size_t bytes) {
  void *p = malloc(bytes);
  if (p) memcpy(p, src, bytes);
  return p;
}

#define BIND_RAM_L(L)                                                            \
  memcpy(r_ln1g[L], TFM_l##L##_ln1_g, sizeof(r_ln1g[L])); LY[L].ln1_g = r_ln1g[L]; \
  memcpy(r_ln1b[L], TFM_l##L##_ln1_b, sizeof(r_ln1b[L])); LY[L].ln1_b = r_ln1b[L]; \
  memcpy(r_ln2g[L], TFM_l##L##_ln2_g, sizeof(r_ln2g[L])); LY[L].ln2_g = r_ln2g[L]; \
  memcpy(r_ln2b[L], TFM_l##L##_ln2_b, sizeof(r_ln2b[L])); LY[L].ln2_b = r_ln2b[L]; \
  memcpy(r_ob[L],   TFM_l##L##_o_b,   sizeof(r_ob[L]));   LY[L].o_b   = r_ob[L];   \
  memcpy(r_fcb[L],  TFM_l##L##_fc_b,  sizeof(r_fcb[L]));  LY[L].fc_b  = r_fcb[L];  \
  memcpy(r_prb[L],  TFM_l##L##_pr_b,  sizeof(r_prb[L]));  LY[L].pr_b  = r_prb[L];  \
  LY[L].q  = (const float(*)[E])  tfm_dup(TFM_l##L##_q,  sizeof(TFM_l##L##_q));    \
  LY[L].k  = (const float(*)[E])  tfm_dup(TFM_l##L##_k,  sizeof(TFM_l##L##_k));    \
  LY[L].v  = (const float(*)[E])  tfm_dup(TFM_l##L##_v,  sizeof(TFM_l##L##_v));    \
  LY[L].o  = (const float(*)[E])  tfm_dup(TFM_l##L##_o,  sizeof(TFM_l##L##_o));    \
  LY[L].fc = (const float(*)[FF]) tfm_dup(TFM_l##L##_fc, sizeof(TFM_l##L##_fc));   \
  LY[L].pr = (const float(*)[E])  tfm_dup(TFM_l##L##_pr, sizeof(TFM_l##L##_pr));   \
  if (!LY[L].q || !LY[L].k || !LY[L].v || !LY[L].o || !LY[L].fc || !LY[L].pr) return 0;

static int tfm_bind_ram(void) {
  BIND_RAM_L(0)
  BIND_RAM_L(1)
  memcpy(r_tok,    TFM_tok_emb, sizeof(r_tok));    TOKE   = (const float(*)[E])r_tok;
  memcpy(r_pos,    TFM_pos_emb, sizeof(r_pos));    POSE   = (const float(*)[E])r_pos;
  memcpy(r_head,   TFM_head,    sizeof(r_head));   HEADW  = (const float(*)[VOC])r_head;
  memcpy(r_head_b, TFM_head_b,  sizeof(r_head_b)); HEAD_B = r_head_b;
  memcpy(r_lnf_g,  TFM_lnf_g,   sizeof(r_lnf_g));  LNF_G  = r_lnf_g;
  memcpy(r_lnf_b,  TFM_lnf_b,   sizeof(r_lnf_b));  LNF_B  = r_lnf_b;
  return 1;
}

// ---- forward --------------------------------------------------------------
static void tfm_layernorm(float *x, const float *g, const float *b) {
  float mu = 0.0f;
  for (int i = 0; i < E; i++) mu += x[i];
  mu /= E;
  float var = 0.0f;
  for (int i = 0; i < E; i++) { float d = x[i] - mu; var += d * d; }
  var /= E;
  const float inv = 1.0f / sqrtf(var + TFM_LN_EPS);
  for (int i = 0; i < E; i++) x[i] = (x[i] - mu) * inv * g[i] + b[i];
}

void tfm_reset(void) { /* nothing to clear: only cache slots 0..t are ever read */ }

// One token at absolute position t (0-based, resets to 0 at each word boundary).
// Writes TFM_VOCAB logits. Reads cache slots 0..t-1, writes slot t.
void tfm_forward(int tok, int t, float *logits) {
  float x[E], h[E], att_out[E], q[E], sc[BS], ff[FF];

  for (int i = 0; i < E; i++) x[i] = TOKE[tok][i] + POSE[t][i];

  for (int L = 0; L < NL; L++) {
    const TfmLayer *ly = &LY[L];

    memcpy(h, x, sizeof(h));
    tfm_layernorm(h, ly->ln1_g, ly->ln1_b);

    // q for the new token; k and v for the new token straight into the cache
    for (int j = 0; j < E; j++) {
      float aq = 0.0f, ak = 0.0f, av = 0.0f;
      for (int k = 0; k < E; k++) {
        const float hk = h[k];
        aq += hk * ly->q[k][j];
        ak += hk * ly->k[k][j];
        av += hk * ly->v[k][j];
      }
      q[j] = aq; kc[L][t][j] = ak; vc[L][t][j] = av;
    }

    for (int hh = 0; hh < NH; hh++) {
      const int off = hh * HS;
      float mx = -1e30f;
      for (int u = 0; u <= t; u++) {            // causal: only keys 0..t
        float s = 0.0f;
        for (int d = 0; d < HS; d++) s += q[off + d] * kc[L][u][off + d];
        s *= TFM_SCALE;
        sc[u] = s;
        if (s > mx) mx = s;
      }
      float sum = 0.0f;
      for (int u = 0; u <= t; u++) { sc[u] = expf(sc[u] - mx); sum += sc[u]; }
      const float isum = 1.0f / sum;
      for (int d = 0; d < HS; d++) {
        float acc = 0.0f;
        for (int u = 0; u <= t; u++) acc += sc[u] * vc[L][u][off + d];
        att_out[off + d] = acc * isum;
      }
    }

    for (int j = 0; j < E; j++) {
      float acc = ly->o_b[j];
      for (int k = 0; k < E; k++) acc += att_out[k] * ly->o[k][j];
      x[j] += acc;
    }

    memcpy(h, x, sizeof(h));
    tfm_layernorm(h, ly->ln2_g, ly->ln2_b);

    for (int j = 0; j < FF; j++) {
      float acc = ly->fc_b[j];
      for (int k = 0; k < E; k++) acc += h[k] * ly->fc[k][j];
      ff[j] = acc > 0.0f ? acc : 0.0f;          // ReLU
    }
    for (int j = 0; j < E; j++) {
      float acc = ly->pr_b[j];
      for (int k = 0; k < FF; k++) acc += ff[k] * ly->pr[k][j];
      x[j] += acc;
    }
  }

  tfm_layernorm(x, LNF_G, LNF_B);
  for (int j = 0; j < VOC; j++) {
    float acc = HEAD_B[j];
    for (int k = 0; k < E; k++) acc += x[k] * HEADW[k][j];
    logits[j] = acc;
  }
}

// Compares tfm_forward against the golden logits baked into the header.
// Returns the max absolute error over all reference positions.
static float tfm_selftest(void) {
  float logits[VOC], worst = 0.0f;
  const int n = (int)(sizeof(TFM_REF_IN) / sizeof(TFM_REF_IN[0]));
  for (int t = 0; t < n; t++) {
    tfm_forward(TFM_REF_IN[t], t, logits);
    for (int j = 0; j < VOC; j++) {
      float d = fabsf(logits[j] - TFM_REF_LOGITS[t][j]);
      if (d > worst) worst = d;
    }
  }
  return worst;
}