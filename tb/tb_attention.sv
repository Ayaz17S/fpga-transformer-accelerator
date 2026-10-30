// tb_attention.sv - checks attention.sv against vectors from the Python reference.
//
// scripts/gen_attention_vectors.py runs attention_fx() over whole sequences:
// at every position t a new q, k and v row is produced and attention runs
// over cache rows 0..t. This testbench replays that exactly: for each t it
// writes row t of k and v into the DUT's KV cache, loads q, runs one command
// and requires a 0-LSB match on all 16 outputs. Rows from earlier positions
// stay in the cache, as they will in the real model.

`timescale 1ns/1ps

module tb_attention;

  import model_pkg::*;

  localparam int SEQS = 8;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic                     k_we, v_we;
  logic [3:0]               wr_slot, wr_dim;
  logic signed [DATA_W-1:0] wr_data;
  logic                     start, busy, done;
  logic [3:0]               pos;
  logic [3:0]               q_addr;
  logic signed [DATA_W-1:0] q_data;
  logic                     o_we;
  logic [3:0]               o_addr;
  logic signed [DATA_W-1:0] o_data;

  attention dut (.*);

  // ---------------- q buffer model and output capture ----------------
  logic signed [DATA_W-1:0] qmem [16];
  logic signed [DATA_W-1:0] omem [16];
  int                       o_writes;

  always_ff @(posedge clk) begin
    q_data <= qmem[q_addr];
    if (o_we) begin
      omem[o_addr] <= o_data;
      o_writes++;
    end
  end

  // ---------------- vectors from Python ----------------
  localparam int ROWS = SEQS * CTX;  // one row of 16 per (sequence, position)
  logic [15:0] vec_q   [ROWS * 16];
  logic [15:0] vec_k   [ROWS * 16];
  logic [15:0] vec_v   [ROWS * 16];
  logic [15:0] vec_out [ROWS * 16];

  int n_runs = 0, n_errors = 0;
  int cycles_at [CTX];

  task automatic write_cache_row(input bit is_k, input int row, input int slot);
    for (int dd = 0; dd < D_MODEL; dd++) begin
      k_we    <= is_k;
      v_we    <= !is_k;
      wr_slot <= slot;
      wr_dim  <= dd;
      wr_data <= is_k ? vec_k[row * 16 + dd] : vec_v[row * 16 + dd];
      @(posedge clk);
    end
    k_we <= 0;
    v_we <= 0;
  endtask

  task automatic run_position(input int seq, input int t);
    int row = seq * CTX + t;
    int cycles = 0;

    write_cache_row(1, row, t);
    write_cache_row(0, row, t);
    for (int dd = 0; dd < D_MODEL; dd++) qmem[dd] = vec_q[row * 16 + dd];
    foreach (omem[dd]) omem[dd] = 'x;
    o_writes = 0;

    @(posedge clk);
    start <= 1;
    pos   <= t;
    @(posedge clk);
    start <= 0;
    pos   <= 'x;  // must be latched
    do begin
      @(posedge clk);
      cycles++;
    end while (!done);
    @(posedge clk);  // let the final output write land

    cycles_at[t] = cycles;
    n_runs++;
    if (o_writes != D_MODEL) begin
      $error("seq %0d t %0d: %0d output writes, expected %0d", seq, t, o_writes, D_MODEL);
      n_errors++;
    end
    for (int dd = 0; dd < D_MODEL; dd++) begin
      if (omem[dd] !== $signed(vec_out[row * 16 + dd])) begin
        $error("seq %0d t %0d: out[%0d] = %0d, expected %0d", seq, t, dd,
               omem[dd], $signed(vec_out[row * 16 + dd]));
        n_errors++;
      end
    end
  endtask

  initial begin
    $readmemh("attn_q.mem", vec_q);
    $readmemh("attn_k.mem", vec_k);
    $readmemh("attn_v.mem", vec_v);
    $readmemh("attn_out.mem", vec_out);
    if ($isunknown(vec_out[ROWS * 16 - 1]))
      $fatal(1, "attention vectors missing: run scripts/gen_attention_vectors.py");

    {k_we, v_we, start} = '0;
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n <= 1;

    $display("[1] %0d sequences x %0d positions", SEQS, CTX);
    for (int seq = 0; seq < SEQS; seq++)
      for (int t = 0; t < CTX; t++)
        run_position(seq, t);

    $display("    cycles per token: t=0 %0d, t=7 %0d, t=15 %0d",
             cycles_at[0], cycles_at[7], cycles_at[15]);
    if (n_errors == 0) $display("PASS: %0d attention runs match Python bit-exactly", n_runs);
    else               $display("FAIL: %0d errors in %0d runs", n_errors, n_runs);
    $finish;
  end

endmodule
