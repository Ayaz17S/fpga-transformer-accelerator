// tb_matvec.sv - self-checking testbench for matvec.sv (+ mac, requant)
//
// The weight and x memories are modelled here as arrays with a 1-cycle
// registered read, exactly like a BRAM. Each test fills them with random
// values, runs one command, captures every y write, and compares against a
// reference computed with 64-bit integers and the contract's requant rule.

`timescale 1ns/1ps

module tb_matvec;

  import tb_pkg::*;  // requant_ref()

  localparam int DATA_W = 16, ACC_W = 48;
  localparam int W_AW = 13, X_AW = 6, Y_AW = 6, DIM_W = 7, SH_W = 6;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic                      start, busy, done;
  logic [W_AW-1:0]           w_base;
  logic [DIM_W-1:0]          n_in, n_out;
  logic [SH_W-1:0]           shift;
  logic [W_AW-1:0]           w_addr;
  logic signed [DATA_W-1:0]  w_data;
  logic [X_AW-1:0]           x_addr;
  logic signed [DATA_W-1:0]  x_data;
  logic                      y_we;
  logic [Y_AW-1:0]           y_addr;
  logic signed [DATA_W-1:0]  y_data;

  matvec #(.DATA_W(DATA_W), .ACC_W(ACC_W), .W_AW(W_AW), .X_AW(X_AW),
           .Y_AW(Y_AW), .DIM_W(DIM_W), .SH_W(SH_W)) dut (.*);

  // ---------------- memory models (1-cycle read latency) ----------------
  logic signed [DATA_W-1:0] wmem [2**W_AW];
  logic signed [DATA_W-1:0] xmem [2**X_AW];
  logic signed [DATA_W-1:0] ymem [2**Y_AW];
  int                       y_writes;

  always_ff @(posedge clk) begin
    w_data <= wmem[w_addr];
    x_data <= xmem[x_addr];
    if (y_we) begin
      ymem[y_addr] <= y_data;
      y_writes++;
    end
  end

  int n_tests = 0, n_errors = 0;

  // Fill W (at base) and x with random values in [-range, range], run one
  // command, and check every output plus the cycle count.
  task automatic run_test(input string name, input int base, input int ni,
                          input int no, input int sh, input int wrange,
                          input int xrange);
    longint exp_y [];
    int     cycles = 0;

    for (int k = 0; k < ni * no; k++)
      wmem[base + k] = $signed($urandom_range(2 * wrange)) - wrange;
    for (int k = 0; k < ni; k++)
      xmem[k] = $signed($urandom_range(2 * xrange)) - xrange;
    foreach (ymem[k]) ymem[k] = 'x;

    exp_y = new[no];
    for (int o = 0; o < no; o++) begin
      longint acc = 0;
      for (int i = 0; i < ni; i++)
        acc += longint'(wmem[base + o * ni + i]) * longint'(xmem[i]);
      exp_y[o] = requant_ref(acc, sh);
    end

    y_writes = 0;
    @(posedge clk);
    start  <= 1;
    w_base <= base;
    n_in   <= ni;
    n_out  <= no;
    shift  <= sh;
    @(posedge clk);
    start  <= 0;
    w_base <= 'x; n_in <= 'x; n_out <= 'x; shift <= 'x;  // must be latched
    do begin
      @(posedge clk);
      cycles++;
    end while (!done);
    @(posedge clk);  // let the final y write land in ymem

    n_tests++;
    if (y_writes != no) begin
      $error("%s: %0d y writes, expected %0d", name, y_writes, no);
      n_errors++;
    end
    for (int o = 0; o < no; o++) begin
      if (ymem[o] !== DATA_W'(exp_y[o])) begin
        $error("%s: y[%0d] = %0d, expected %0d", name, o, ymem[o], exp_y[o]);
        n_errors++;
      end
    end
    $display("  %-24s %2d x %2d  shift %2d  %4d cycles (%0d MACs)",
             name, no, ni, sh, cycles, ni * no);
  endtask

  initial begin
    start = 0;
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n <= 1;

    $display("[1] the model's layer shapes, typical value ranges");
    run_test("wq (16x16)",      0,    16, 16, 16, 4096, 8192);
    run_test("fc1 (64x16)",     256,  16, 64, 16, 4096, 8192);
    run_test("fc2 (16x64)",     1280, 64, 16, 16, 4096, 8192);
    run_test("lm_head (27x16)", 2304, 16, 27, 14, 4096, 8192);

    $display("[2] edge shapes");
    run_test("1x1",             5000, 1,  1,  4,  4096, 8192);
    run_test("1x64",            5000, 64, 1,  16, 4096, 8192);
    run_test("64x1",            5000, 1,  64, 4,  4096, 8192);
    run_test("64x64",           0,    64, 64, 16, 4096, 8192);

    $display("[3] saturation: full-range values, small shift");
    run_test("saturate",        100,  16, 16, 8,  32768, 32767);
    run_test("no shift",        100,  4,  4,  0,  8,     8);

    $display("[4] random shapes, bases and shifts");
    for (int t = 0; t < 40; t++)
      run_test($sformatf("random %0d", t), $urandom_range(0, 4000),
               $urandom_range(1, 64), $urandom_range(1, 64),
               $urandom_range(10, 20), 4096, 8192);

    if (n_errors == 0) $display("PASS: %0d matvec commands checked", n_tests);
    else               $display("FAIL: %0d errors in %0d commands", n_errors, n_tests);
    $finish;
  end

endmodule
