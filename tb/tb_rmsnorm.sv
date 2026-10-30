// tb_rmsnorm.sv - checks rmsnorm.sv against vectors from the Python reference.
//
// scripts/gen_rmsnorm_vectors.py runs rmsnorm_fx() from scripts/fixedpoint.py
// and writes inputs, output formats and expected outputs as .mem files. This
// testbench replays every vector and requires a 0-LSB match: the first check
// of RTL against the Python golden model rather than a SystemVerilog copy.

`timescale 1ns/1ps

module tb_rmsnorm;

  import model_pkg::*;

  localparam int MAX_VECS = 1024;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic                     start, busy, done;
  logic [3:0]               out_frac;
  logic [3:0]               x_addr;
  logic signed [DATA_W-1:0] x_data;
  logic                     h_we;
  logic [3:0]               h_addr;
  logic signed [DATA_W-1:0] h_data;

  rmsnorm dut (.*);

  // ---------------- x buffer model and h capture ----------------
  logic signed [DATA_W-1:0] xmem [D_MODEL];
  logic signed [DATA_W-1:0] hmem [D_MODEL];
  int                       h_writes;

  always_ff @(posedge clk) begin
    x_data <= xmem[x_addr];
    if (h_we) begin
      hmem[h_addr] <= h_data;
      h_writes++;
    end
  end

  // ---------------- vectors from Python ----------------
  logic [15:0] vec_x    [MAX_VECS * D_MODEL];
  logic [15:0] vec_frac [MAX_VECS];
  logic [15:0] vec_h    [MAX_VECS * D_MODEL];

  int n_vecs = 0, n_errors = 0, cycles_min = 1 << 30, cycles_max = 0;

  task automatic run_vector(input int n);
    int cycles = 0;

    for (int d = 0; d < D_MODEL; d++) xmem[d] = vec_x[n * D_MODEL + d];
    foreach (hmem[d]) hmem[d] = 'x;
    h_writes = 0;

    @(posedge clk);
    start    <= 1;
    out_frac <= vec_frac[n][3:0];
    @(posedge clk);
    start    <= 0;
    out_frac <= 'x;  // must be latched
    do begin
      @(posedge clk);
      cycles++;
    end while (!done);
    @(posedge clk);  // let the final h write land

    cycles_min = (cycles < cycles_min) ? cycles : cycles_min;
    cycles_max = (cycles > cycles_max) ? cycles : cycles_max;

    if (h_writes != D_MODEL) begin
      $error("vector %0d: %0d h writes, expected %0d", n, h_writes, D_MODEL);
      n_errors++;
    end
    for (int d = 0; d < D_MODEL; d++) begin
      if (hmem[d] !== $signed(vec_h[n * D_MODEL + d])) begin
        $error("vector %0d (Q.%0d): h[%0d] = %0d, expected %0d", n, vec_frac[n],
               d, hmem[d], $signed(vec_h[n * D_MODEL + d]));
        n_errors++;
      end
    end
  endtask

  initial begin
    $readmemh("rmsnorm_x.mem", vec_x);
    $readmemh("rmsnorm_frac.mem", vec_frac);
    $readmemh("rmsnorm_h.mem", vec_h);
    while (n_vecs < MAX_VECS && !$isunknown(vec_frac[n_vecs])) n_vecs++;
    if (n_vecs == 0) $fatal(1, "no vectors: run scripts/gen_rmsnorm_vectors.py");

    start = 0;
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n <= 1;

    $display("[1] %0d vectors from scripts/gen_rmsnorm_vectors.py", n_vecs);
    for (int n = 0; n < n_vecs; n++) run_vector(n);

    if (n_errors == 0)
      $display("PASS: %0d vectors match Python bit-exactly, %0d-%0d cycles each",
               n_vecs, cycles_min, cycles_max);
    else
      $display("FAIL: %0d errors in %0d vectors", n_errors, n_vecs);
    $finish;
  end

endmodule
