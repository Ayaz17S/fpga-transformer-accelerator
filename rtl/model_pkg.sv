// model_pkg.sv - model dimensions and the weight ROM memory map.
//
// Every RTL module that needs a dimension or a weight address imports this
// package, so the numbers live in exactly one place on the RTL side.
// scripts/gen_placeholder_weights.py writes the ROM image in this same order;
// keep the two in sync.

`timescale 1ns/1ps

package model_pkg;

  // ---------------- model dimensions ----------------
  localparam int D_MODEL  = 16;   // embedding width
  localparam int N_HEAD   = 4;
  localparam int HEAD_DIM = 4;
  localparam int CTX      = 16;   // context length
  localparam int VOCAB    = 27;
  localparam int D_FF     = 64;   // MLP hidden width

  // ---------------- datapath widths ----------------
  localparam int DATA_W = 16;
  localparam int ACC_W  = 48;
  localparam int W_AW   = 13;     // weight ROM address bits (8192 words)

  // ---------------- weight ROM map (words, row-major [out][in]) ----------------
  localparam int WTE_BASE  = 0;                            // VOCAB   x D_MODEL
  localparam int WPE_BASE  = WTE_BASE + VOCAB   * D_MODEL; // CTX     x D_MODEL
  localparam int WQ_BASE   = WPE_BASE + CTX     * D_MODEL; // D_MODEL x D_MODEL
  localparam int WK_BASE   = WQ_BASE  + D_MODEL * D_MODEL;
  localparam int WV_BASE   = WK_BASE  + D_MODEL * D_MODEL;
  localparam int WO_BASE   = WV_BASE  + D_MODEL * D_MODEL;
  localparam int FC1_BASE  = WO_BASE  + D_MODEL * D_MODEL; // D_FF    x D_MODEL
  localparam int FC2_BASE  = FC1_BASE + D_FF    * D_MODEL; // D_MODEL x D_FF
  localparam int LM_BASE   = FC2_BASE + D_MODEL * D_FF;    // VOCAB   x D_MODEL
  localparam int N_WEIGHTS = LM_BASE  + VOCAB   * D_MODEL; // = 4192

  // ---------------- requantize shifts ----------------
  // Placeholders until the ML side generates quant_params.svh.
  localparam int EMB_SHIFT       = 4;   // wte + wpe: Q2.14 -> Q6.10
  localparam int SHIFT_FROM_NORM = 16;  // Q2.14 weights x Q4.12 normed input -> Q6.10
  localparam int SHIFT_FROM_ACT  = 14;  // Q2.14 weights x Q6.10 activation   -> Q6.10
  localparam int NORM0_FRAC      = 10;  // first rmsnorm feeds the residual stream: Q6.10
  localparam int NORM_FRAC       = 12;  // block rmsnorms: Q4.12

  // ---------------- RMSNorm (see scripts/fixedpoint.py) ----------------
  localparam int RMS_EPS_RAW    = 10;  // round(1e-5 * 2^20) for Q6.10 inputs
  localparam int RMS_LUT_BITS   = 10;  // 1024-entry rsqrt table
  localparam int RMS_SHIFT_BASE = 13;  // 15 (LUT scale) - log2(sqrt(D_MODEL))

  // ---------------- softmax (see scripts/fixedpoint.py) ----------------
  localparam int EXP_LUT_BITS  = 10;  // 1024-entry exp table, step 1/64
  localparam int EXP_IDX_SHIFT = 4;   // Q6.10 difference -> table index
  localparam int SM_RECIP_BITS = 30;  // R = floor(2^30 / sum)
  localparam int SM_PROB_SHIFT = 16;  // e * R -> Q2.14

  // ---------------- attention ----------------
  localparam int ATT_SCORE_SHIFT = 11;  // q.k Q12.20 -> Q6.10, plus x0.5 (1/sqrt(HEAD_DIM))
  localparam int ATT_OUT_SHIFT   = 14;  // p.v Q8.24 -> Q6.10

endpackage
