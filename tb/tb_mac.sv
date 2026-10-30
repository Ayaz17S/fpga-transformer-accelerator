// tb_mac.sv - self-checking testbench for mac.sv
//
// The driver sends dot-product bursts and, for each burst, pushes the
// expected sum (computed with 64-bit integers) into a queue. The checker pops
// one expected value every time the DUT raises out_valid and compares.

`timescale 1ns/1ps

module tb_mac;

  localparam int A_W = 16, B_W = 16, ACC_W = 48;

  logic                    clk = 0;
  logic                    rst_n;
  logic                    in_valid, in_first, in_last;
  logic signed [A_W-1:0]   a;
  logic signed [B_W-1:0]   b;
  logic                    out_valid;
  logic signed [ACC_W-1:0] acc;

  mac #(.A_W(A_W), .B_W(B_W), .ACC_W(ACC_W)) dut (.*);

  always #5 clk = ~clk;  // 100 MHz

  // ---------------- scoreboard ----------------
  longint expected_q[$];
  int     n_checked = 0, n_errors = 0;

  always @(posedge clk) begin
    if (out_valid) begin
      if (expected_q.size() == 0) begin
        $error("out_valid with no expected result pending");
        n_errors++;
      end else begin
        automatic longint exp_v = expected_q.pop_front();
        if (acc !== ACC_W'(exp_v)) begin
          $error("result %0d: got %0d, expected %0d", n_checked, acc, exp_v);
          n_errors++;
        end
        n_checked++;
      end
    end
  end

  // ---------------- driver ----------------
  task automatic idle();
    in_valid <= 0; in_first <= 0; in_last <= 0;
    a <= 'x; b <= 'x;  // data must not matter while in_valid is low
    @(posedge clk);
  endtask

  // Send one dot product of the given vectors. bubble_pct = chance (0-100)
  // of inserting an idle cycle before each element.
  task automatic send_dot(input logic signed [A_W-1:0] av[],
                          input logic signed [B_W-1:0] bv[],
                          input int bubble_pct = 0);
    longint sum = 0;
    for (int i = 0; i < av.size(); i++) begin
      while ($urandom_range(99) < bubble_pct) idle();
      in_valid <= 1;
      in_first <= (i == 0);
      in_last  <= (i == av.size() - 1);
      a <= av[i];
      b <= bv[i];
      sum += longint'(av[i]) * longint'(bv[i]);
      @(posedge clk);
    end
    expected_q.push_back(sum);
  endtask

  task automatic send_random(input int len, input int bubble_pct = 0);
    logic signed [A_W-1:0] av[] = new[len];
    logic signed [B_W-1:0] bv[] = new[len];
    foreach (av[i]) begin
      av[i] = $urandom();
      bv[i] = $urandom();
    end
    send_dot(av, bv, bubble_pct);
  endtask

  task automatic send_const(input int len, input int va, input int vb);
    logic signed [A_W-1:0] av[] = new[len];
    logic signed [B_W-1:0] bv[] = new[len];
    foreach (av[i]) begin
      av[i] = va;
      bv[i] = vb;
    end
    send_dot(av, bv);
  endtask

  // ---------------- tests ----------------
  initial begin
    rst_n = 0;
    idle(); idle(); idle();
    rst_n <= 1;
    idle();

    $display("[1] single-element dot products, back to back");
    for (int i = 0; i < 8; i++) send_random(1);

    $display("[2] back-to-back bursts of 16 and 64 (the model's dot lengths)");
    for (int i = 0; i < 4; i++) begin
      send_random(16);
      send_random(64);
    end

    $display("[3] extremes: largest positive and negative sums of length 64");
    send_const(64, -32768, -32768);  // +2^36: needs 38 bits
    send_const(64, -32768,  32767);  // most negative sum
    send_const(64,  32767,  32767);

    $display("[4] random lengths 1..64 with random bubbles");
    for (int i = 0; i < 200; i++) send_random($urandom_range(1, 64), 30);

    idle();
    repeat (5) idle();  // drain the 3-stage pipeline

    if (expected_q.size() != 0) begin
      $error("%0d results never came out", expected_q.size());
      n_errors++;
    end

    if (n_errors == 0) $display("PASS: %0d dot products checked", n_checked);
    else               $display("FAIL: %0d errors in %0d checks", n_errors, n_checked);
    $finish;
  end

endmodule
