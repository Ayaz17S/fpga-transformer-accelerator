// tb_gpt_core.sv - end-to-end check of gpt_core.sv against GptFx in Python.
//
// scripts/gen_model_vectors.py runs whole sequences of 16 tokens through the
// bit-exact Python model. This testbench feeds the same tokens at positions
// 0..15 and, after every step, compares with 0-LSB tolerance:
//   - the residual stream x after the MLP block (read hierarchically)
//   - all 27 logits (through the logit read port)
//   - the argmax next token

`timescale 1ns/1ps

module tb_gpt_core;

  import model_pkg::*;

  localparam int SEQS  = 6;
  localparam int STEPS = SEQS * CTX;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic                     start, busy, done;
  logic [4:0]               token, next_token;
  logic [3:0]               pos;
  logic [4:0]               logit_addr;
  logic signed [DATA_W-1:0] logit_data;

  gpt_core dut (.*);

  // ---------------- vectors from Python ----------------
  logic [15:0] vec_tokens [STEPS];
  logic [15:0] vec_x      [STEPS * D_MODEL];
  logic [15:0] vec_logits [STEPS * VOCAB];
  logic [15:0] vec_next   [STEPS];

  int n_steps = 0, n_errors = 0, n_first_err = -1;
  int cycles = 0;

  task automatic check_step(input int s);
    int errs = 0;
    for (int d = 0; d < D_MODEL; d++)
      if (dut.x_buf[d] !== $signed(vec_x[s * D_MODEL + d])) begin
        if (errs < 4) $error("step %0d: x[%0d] = %0d, expected %0d", s, d,
                             dut.x_buf[d], $signed(vec_x[s * D_MODEL + d]));
        errs++;
      end
    for (int v = 0; v < VOCAB; v++) begin
      logit_addr = v;
      #1;
      if (logit_data !== $signed(vec_logits[s * VOCAB + v])) begin
        if (errs < 4) $error("step %0d: logit[%0d] = %0d, expected %0d", s, v,
                             logit_data, $signed(vec_logits[s * VOCAB + v]));
        errs++;
      end
    end
    if (next_token !== vec_next[s][4:0]) begin
      $error("step %0d: next token %0d, expected %0d", s, next_token, vec_next[s]);
      errs++;
    end
    if (errs > 0 && n_first_err < 0) n_first_err = s;
    n_errors += errs;
  endtask

  initial begin
    $readmemh("model_tokens.mem", vec_tokens);
    $readmemh("model_x.mem", vec_x);
    $readmemh("model_logits.mem", vec_logits);
    $readmemh("model_next.mem", vec_next);
    if ($isunknown(vec_next[STEPS - 1]))
      $fatal(1, "model vectors missing: run scripts/gen_model_vectors.py");

    start = 0;
    logit_addr = 0;
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n <= 1;

    $display("[1] %0d sequences x %0d tokens through the whole model", SEQS, CTX);
    for (int s = 0; s < STEPS; s++) begin
      @(posedge clk);
      start <= 1;
      token <= vec_tokens[s][4:0];
      pos   <= s % CTX;
      @(posedge clk);
      start <= 0;
      token <= 'x; pos <= 'x;  // must be latched
      cycles = 0;
      do begin
        @(posedge clk);
        cycles++;
      end while (!done);
      if (s % CTX == 0 || s % CTX == CTX - 1)
        $display("    seq %0d pos %2d: %0d cycles", s / CTX, s % CTX, cycles);
      check_step(s);
      n_steps++;
    end

    if (n_errors == 0)
      $display("PASS: %0d token steps match Python bit-exactly (x, 27 logits, next token)", n_steps);
    else
      $display("FAIL: %0d errors in %0d steps, first at step %0d", n_errors, n_steps, n_first_err);
    $finish;
  end

endmodule
