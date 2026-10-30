// requant.sv - round half up, arithmetic shift right, saturate.
//
// Bit-exact twin of the contract's Python requant():
//   acc = (acc + (1 << (shift-1))) >> shift   (shift > 0)
//   clamp to [-2^(OUT_W-1), 2^(OUT_W-1)-1]
// Only right shifts (shift >= 0) are supported: every dot product in this
// model has more fraction bits than its output format.
//
// Two pipeline stages (latency 2). Doing the 48-bit add, the variable shift
// and the saturation compare in one cycle took ~10.9 ns on a 7-series part,
// which misses 100 MHz, so the work is split:
//   stage 1: biased = in + rounding bias
//   stage 2: out    = saturate(biased >>> shift)

`timescale 1ns/1ps

module requant #(
  parameter int IN_W  = 48,
  parameter int OUT_W = 16,
  parameter int SH_W  = 6
) (
  input  logic                    clk,
  input  logic                    rst_n,     // synchronous, active low

  input  logic                    in_valid,
  input  logic signed [IN_W-1:0]  in,
  input  logic        [SH_W-1:0]  shift,     // 0 .. IN_W-1

  output logic                    out_valid,
  output logic signed [OUT_W-1:0] out
);

  localparam logic signed [IN_W-1:0] ONE     = 1;
  localparam logic signed [IN_W-1:0] OUT_MAX =  (2**(OUT_W-1)) - 1;
  localparam logic signed [IN_W-1:0] OUT_MIN = -(2**(OUT_W-1));

  // ---------------- stage 1: add the rounding bias ----------------
  logic signed [IN_W-1:0] bias, biased;
  logic        [SH_W-1:0] shift1;
  logic                   v1;

  assign bias = (shift == 0) ? '0 : (ONE <<< (shift - 1));

  always_ff @(posedge clk) begin
    biased <= in + bias;
    shift1 <= shift;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) v1 <= 1'b0;
    else        v1 <= in_valid;
  end

  // ---------------- stage 2: shift and saturate ----------------
  logic signed [IN_W-1:0] shifted;

  assign shifted = biased >>> shift1;

  always_ff @(posedge clk) begin
    if      (shifted > OUT_MAX) out <= OUT_MAX[OUT_W-1:0];
    else if (shifted < OUT_MIN) out <= OUT_MIN[OUT_W-1:0];
    else                        out <= shifted[OUT_W-1:0];
  end

  always_ff @(posedge clk) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= v1;
  end

endmodule
