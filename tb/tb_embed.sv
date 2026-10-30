// tb_embed.sv - self-checking testbench for embed.sv + weight_rom.sv
//
// Uses the real ROM loaded from weights_all.mem. The testbench loads the
// same file into its own array for the reference, then runs every
// (token, position) pair and checks all 16 outputs and the cycle count.

`timescale 1ns/1ps

module tb_embed;

  import model_pkg::*;
  import tb_pkg::*;  // requant_ref()

  localparam string MEM_FILE = "weights_all.mem";

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic                     start, busy, done;
  logic [4:0]               token;
  logic [3:0]               pos;
  logic [W_AW-1:0]          wa_addr, wb_addr;
  logic signed [DATA_W-1:0] wa_data, wb_data;
  logic                     x_we;
  logic [3:0]               x_addr;
  logic signed [DATA_W-1:0] x_data;

  weight_rom #(.DATA_W(DATA_W), .AW(W_AW), .DEPTH(N_WEIGHTS),
               .INIT_FILE(MEM_FILE)) u_rom (
    .clk   (clk),
    .a_addr(wa_addr), .a_data(wa_data),
    .b_addr(wb_addr), .b_data(wb_data)
  );

  embed dut (.*);

  // ---------------- capture x writes ----------------
  logic signed [DATA_W-1:0] xmem [D_MODEL];
  int                       x_writes;

  always_ff @(posedge clk) begin
    if (x_we) begin
      xmem[x_addr] <= x_data;
      x_writes++;
    end
  end

  // ---------------- reference ----------------
  logic signed [DATA_W-1:0] ref_w [N_WEIGHTS];
  int n_tests = 0, n_errors = 0;

  task automatic run_test(input int tok, input int p);
    int cycles = 0;

    foreach (xmem[d]) xmem[d] = 'x;
    x_writes = 0;

    @(posedge clk);
    start <= 1;
    token <= tok;
    pos   <= p;
    @(posedge clk);
    start <= 0;
    token <= 'x; pos <= 'x;  // must be latched
    do begin
      @(posedge clk);
      cycles++;
    end while (!done);
    @(posedge clk);  // let the final x write land

    n_tests++;
    if (x_writes != D_MODEL) begin
      $error("tok %0d pos %0d: %0d x writes, expected %0d", tok, p, x_writes, D_MODEL);
      n_errors++;
    end
    if (cycles != D_MODEL + 3) begin
      $error("tok %0d pos %0d: took %0d cycles, expected %0d", tok, p, cycles, D_MODEL + 3);
      n_errors++;
    end
    for (int d = 0; d < D_MODEL; d++) begin
      longint sum = longint'(ref_w[WTE_BASE + tok * D_MODEL + d])
                  + longint'(ref_w[WPE_BASE + p   * D_MODEL + d]);
      longint exp_v = requant_ref(sum, EMB_SHIFT);
      if (xmem[d] !== DATA_W'(exp_v)) begin
        $error("tok %0d pos %0d: x[%0d] = %0d, expected %0d", tok, p, d, xmem[d], exp_v);
        n_errors++;
      end
    end
  endtask

  initial begin
    $readmemh(MEM_FILE, ref_w);
    if ($isunknown(ref_w[N_WEIGHTS - 1])) begin
      $fatal(1, "%s missing or shorter than %0d words", MEM_FILE, N_WEIGHTS);
    end

    start = 0;
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n <= 1;

    $display("[1] every token at every position (%0d x %0d)", VOCAB, CTX);
    for (int tok = 0; tok < VOCAB; tok++)
      for (int p = 0; p < CTX; p++)
        run_test(tok, p);

    if (n_errors == 0) $display("PASS: %0d embeddings checked, %0d cycles each", n_tests, D_MODEL + 3);
    else               $display("FAIL: %0d errors in %0d embeddings", n_errors, n_tests);
    $finish;
  end

endmodule
