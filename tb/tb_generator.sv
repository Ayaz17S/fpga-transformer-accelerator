// tb_generator.sv - checks generator.sv against generate_fx() in Python.
//
// scripts/gen_model_vectors.py generates a sequence from every possible start
// token (0..26). This testbench runs the same 27 generations on the hardware
// and compares the length and every generated token.

`timescale 1ns/1ps

module tb_generator;

  import model_pkg::*;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic       start, busy, done;
  logic [4:0] start_token;
  logic [4:0] gen_len;
  logic [3:0] gen_addr;
  logic [4:0] gen_token;

  generator dut (.*);

  logic [15:0] vec_len    [VOCAB];
  logic [15:0] vec_tokens [VOCAB * CTX];

  int n_errors = 0, n_bos = 0, n_full = 0;
  longint total_cycles = 0;

  // the generated tokens as letters, read straight from the DUT's buffer
  function automatic string as_text(input int len);
    string txt = "";
    for (int i = 0; i < len; i++)
      txt = {txt, string'(8'(dut.gen_tokens[i] + 97))};  // token 0..25 -> 'a'..'z'
    return txt;
  endfunction

  initial begin
    $readmemh("gen_len.mem", vec_len);
    $readmemh("gen_tokens.mem", vec_tokens);
    if ($isunknown(vec_len[VOCAB - 1]))
      $fatal(1, "generation vectors missing: run scripts/gen_model_vectors.py");

    start = 0;
    gen_addr = 0;
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n <= 1;

    $display("[1] generate from every start token 0..%0d", VOCAB - 1);
    for (int s = 0; s < VOCAB; s++) begin
      automatic int cycles = 0;
      @(posedge clk);
      start       <= 1;
      start_token <= s;
      @(posedge clk);
      start       <= 0;
      start_token <= 'x;  // must be latched
      do begin
        @(posedge clk);
        cycles++;
      end while (!done);
      total_cycles += cycles;

      if (gen_len !== vec_len[s][4:0]) begin
        $error("start %0d: length %0d, expected %0d", s, gen_len, vec_len[s]);
        n_errors++;
      end else begin
        for (int i = 0; i < gen_len; i++) begin
          gen_addr = i;
          #1;
          if (gen_token !== vec_tokens[s * CTX + i][4:0]) begin
            $error("start %0d: token %0d = %0d, expected %0d", s, i, gen_token,
                   vec_tokens[s * CTX + i]);
            n_errors++;
          end
        end
      end
      if (gen_len < CTX) n_bos++; else n_full++;
      if (s < 3 || s == BOS)
        $display("    start %2d -> \"%s\" (%0d tokens, %0d cycles)", s,
                 as_text(gen_len), gen_len, cycles);
    end

    $display("    %0d stopped on BOS, %0d filled the context; %0d cycles total",
             n_bos, n_full, total_cycles);
    if (n_errors == 0) $display("PASS: %0d generations match Python token for token", VOCAB);
    else               $display("FAIL: %0d errors in %0d generations", n_errors, VOCAB);
    $finish;
  end

endmodule
