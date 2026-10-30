// mac.sv - signed multiply-accumulate unit for dot products.
//
// Streams one (a, b) pair per cycle and accumulates a*b into a wide
// accumulator. One dot product = a burst of pairs marked by in_first on the
// first pair and in_last on the last pair (both high for a 1-element dot
// product). Bursts can follow back-to-back with no idle cycle between them,
// and in_valid may drop mid-burst (bubbles are simply skipped).
//
// Pipeline (latency 3), laid out to map onto one DSP48E1:
//   stage 1: register a, b          -> DSP AREG / BREG
//   stage 2: m = a * b              -> DSP MREG
//   stage 3: acc = first ? m : acc+m -> DSP PREG (post-adder as accumulator)
//
// out_valid pulses for one cycle when acc holds a finished dot product.
// acc is raw: requantization to 16 bits happens downstream (see contract doc).

`timescale 1ns/1ps

module mac #(
  parameter int A_W   = 16,
  parameter int B_W   = 16,
  parameter int ACC_W = 48
) (
  input  logic                    clk,
  input  logic                    rst_n,     // synchronous, active low

  input  logic                    in_valid,
  input  logic                    in_first,
  input  logic                    in_last,
  input  logic signed [A_W-1:0]   a,
  input  logic signed [B_W-1:0]   b,

  output logic                    out_valid,
  output logic signed [ACC_W-1:0] acc
);

  // ---------------- stage 1: input registers ----------------
  logic signed [A_W-1:0] a_r;
  logic signed [B_W-1:0] b_r;
  logic                  v1, first1, last1;

  // Data registers have no reset: DSP48 data registers absorb best without
  // one, and their contents don't matter while the valid flag is low.
  always_ff @(posedge clk) begin
    a_r <= a;
    b_r <= b;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v1 <= 1'b0;
    end else begin
      v1 <= in_valid;
    end
    first1 <= in_first;
    last1  <= in_last;
  end

  // ---------------- stage 2: multiply ----------------
  logic signed [A_W+B_W-1:0] m_r;
  logic                      v2, first2, last2;

  always_ff @(posedge clk) begin
    m_r <= a_r * b_r;  // both operands signed -> signed multiply
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v2 <= 1'b0;
    end else begin
      v2 <= v1;
    end
    first2 <= first1;
    last2  <= last1;
  end

  // ---------------- stage 3: accumulate ----------------
  // first2 loads the product instead of adding it, so a new dot product
  // starts without a separate clear cycle.
  always_ff @(posedge clk) begin
    if (v2) begin
      if (first2) acc <= ACC_W'(m_r);        // sign-extends (m_r is signed)
      else        acc <= acc + ACC_W'(m_r);
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= v2 && last2;
  end

endmodule
