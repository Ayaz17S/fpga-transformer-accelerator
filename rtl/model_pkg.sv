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
  localparam int EMB_SHIFT = 4;   // wte + wpe: Q2.14 -> Q6.10

endpackage
