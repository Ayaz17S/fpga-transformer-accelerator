// tb_softmax.sv - checks softmax.sv against vectors from the Python reference.
//
// scripts/gen_attention_vectors.py runs softmax_fx() from
// scripts/fixedpoint.py and writes lengths, scores and expected
// probabilities. Every vector must match with 0-LSB tolerance.

`timescale 1ns/1ps

module tb_softmax;

  import model_pkg::*;

  localparam int MAX_VECS = 1024;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic                     start, busy, done;
  logic [4:0]               n;
  logic [3:0]               s_addr;
  logic signed [DATA_W-1:0] s_data;
  logic                     p_we;
  logic [3:0]               p_addr;
  logic signed [DATA_W-1:0] p_data;

  softmax dut (.*);

  // ---------------- score buffer model and p capture ----------------
  logic signed [DATA_W-1:0] smem [16];
  logic signed [DATA_W-1:0] pmem [16];
  int                       p_writes;

  always_ff @(posedge clk) begin
    s_data <= smem[s_addr];
    if (p_we) begin
      pmem[p_addr] <= p_data;
      p_writes++;
    end
  end

  // ---------------- vectors from Python ----------------
  logic [15:0] vec_n [MAX_VECS];
  logic [15:0] vec_s [MAX_VECS * 16];
  logic [15:0] vec_p [MAX_VECS * 16];

  int n_vecs = 0, n_errors = 0, cycles_min = 1 << 30, cycles_max = 0;

  task automatic run_vector(input int v);
    int len = vec_n[v];
    int cycles = 0;

    for (int j = 0; j < 16; j++) smem[j] = vec_s[v * 16 + j];
    foreach (pmem[j]) pmem[j] = 'x;
    p_writes = 0;

    @(posedge clk);
    start <= 1;
    n     <= len;
    @(posedge clk);
    start <= 0;
    n     <= 'x;  // must be latched
    do begin
      @(posedge clk);
      cycles++;
    end while (!done);
    @(posedge clk);  // let the final p write land

    cycles_min = (cycles < cycles_min) ? cycles : cycles_min;
    cycles_max = (cycles > cycles_max) ? cycles : cycles_max;

    if (p_writes != len) begin
      $error("vector %0d: %0d p writes, expected %0d", v, p_writes, len);
      n_errors++;
    end
    for (int j = 0; j < len; j++) begin
      if (pmem[j] !== $signed(vec_p[v * 16 + j])) begin
        $error("vector %0d (n=%0d): p[%0d] = %0d, expected %0d", v, len, j,
               pmem[j], $signed(vec_p[v * 16 + j]));
        n_errors++;
      end
    end
  endtask

  initial begin
    $readmemh("softmax_n.mem", vec_n);
    $readmemh("softmax_s.mem", vec_s);
    $readmemh("softmax_p.mem", vec_p);
    while (n_vecs < MAX_VECS && !$isunknown(vec_n[n_vecs])) n_vecs++;
    if (n_vecs == 0) $fatal(1, "no vectors: run scripts/gen_attention_vectors.py");

    start = 0;
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n <= 1;

    $display("[1] %0d vectors from scripts/gen_attention_vectors.py", n_vecs);
    for (int v = 0; v < n_vecs; v++) run_vector(v);

    if (n_errors == 0)
      $display("PASS: %0d vectors match Python bit-exactly, %0d-%0d cycles each",
               n_vecs, cycles_min, cycles_max);
    else
      $display("FAIL: %0d errors in %0d vectors", n_errors, n_vecs);
    $finish;
  end

endmodule
