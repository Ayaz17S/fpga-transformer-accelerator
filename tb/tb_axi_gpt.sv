// tb_axi_gpt.sv - drives axi_gpt_top over AXI4-Lite the way the ARM will.
//
// axi_write / axi_read below are a tiny bus-functional model: they toggle
// VALID / READY exactly as an AXI master does. The test then uses only
// register accesses, like the PYNQ host script:
//   - ID and reset values
//   - generations from several start tokens, compared with the Python
//     vectors (length and every token read back through TOKENS[i])
//   - a write whose data arrives after its address (the slave must wait)
//   - a second start written while busy (must be ignored)

`timescale 1ns/1ps

module tb_axi_gpt;

  import model_pkg::*;

  localparam int CTRL = 'h00, STATUS = 'h04, START_TOKEN = 'h08,
                 GEN_LEN = 'h0C, CYCLES = 'h10, ID = 'h14, TOKENS = 'h40;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic [6:0]  awaddr, araddr;
  logic        awvalid, awready, wvalid, wready, bvalid, bready;
  logic        arvalid, arready, rvalid, rready;
  logic [31:0] wdata, rdata;
  logic [3:0]  wstrb;
  logic [1:0]  bresp, rresp;

  axi_gpt_top dut (
    .s_axi_aclk(clk), .s_axi_aresetn(rst_n),
    .s_axi_awaddr(awaddr), .s_axi_awprot(3'b0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arprot(3'b0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready)
  );

  // ---------------- AXI4-Lite master tasks ----------------
  // A handshake happens on a rising edge where VALID and READY are both 1.
  // The tasks look at READY on the falling edge, so they know the coming
  // rising edge completes the handshake, then drop VALID after it.

  task automatic axi_write(input int addr, input int data, input int data_delay = 0);
    awaddr  <= addr;
    awvalid <= 1;
    repeat (data_delay) @(posedge clk);  // optionally send the data later
    wdata   <= data;
    wstrb   <= 4'hF;
    wvalid  <= 1;
    do @(negedge clk); while (!(awready && wready));
    @(posedge clk);
    awvalid <= 0;
    wvalid  <= 0;
    bready  <= 1;
    do @(negedge clk); while (!bvalid);
    if (bresp !== 2'b00) $error("write 0x%02h: bresp %b", addr, bresp);
    @(posedge clk);
    bready <= 0;
  endtask

  task automatic axi_read(input int addr, output int data);
    araddr  <= addr;
    arvalid <= 1;
    do @(negedge clk); while (!arready);
    @(posedge clk);
    arvalid <= 0;
    rready  <= 1;
    do @(negedge clk); while (!rvalid);
    data = rdata;
    if (rresp !== 2'b00) $error("read 0x%02h: rresp %b", addr, rresp);
    @(posedge clk);
    rready <= 0;
  endtask

  // ---------------- checks ----------------
  logic [15:0] vec_len    [VOCAB];
  logic [15:0] vec_tokens [VOCAB * CTX];
  int n_errors = 0;

  task automatic expect_eq(input string what, input int got, input int expected);
    if (got !== expected) begin
      $error("%s: got %0d (0x%0h), expected %0d (0x%0h)", what, got, got, expected, expected);
      n_errors++;
    end
  endtask

  // Run one generation through registers only; returns the cycle count.
  task automatic generate_and_check(input int start_tok, input int data_delay,
                                    input bit restart_while_busy, output int cycles);
    int status, len, tok, polls = 0;
    bit saw_busy = 0;

    axi_write(START_TOKEN, start_tok, data_delay);
    axi_write(CTRL, 1);
    if (restart_while_busy) axi_write(CTRL, 1);  // must be ignored
    do begin
      axi_read(STATUS, status);
      if (status[0]) saw_busy = 1;
      polls++;
    end while (!status[1]);

    axi_read(GEN_LEN, len);
    axi_read(CYCLES, cycles);
    expect_eq($sformatf("start %0d: GEN_LEN", start_tok), len, vec_len[start_tok]);
    for (int i = 0; i < len; i++) begin
      axi_read(TOKENS + 4 * i, tok);
      expect_eq($sformatf("start %0d: TOKENS[%0d]", start_tok, i), tok,
                vec_tokens[start_tok * CTX + i]);
    end
    if (!saw_busy) begin
      $error("start %0d: STATUS.busy never seen", start_tok);
      n_errors++;
    end
    $display("    start %2d: %2d tokens, %6d cycles, %0d status polls%s", start_tok, len,
             cycles, polls, restart_while_busy ? " (second start ignored)" : "");
  endtask

  initial begin
    int v, cycles_a, cycles_b;

    $readmemh("gen_len.mem", vec_len);
    $readmemh("gen_tokens.mem", vec_tokens);
    if ($isunknown(vec_len[VOCAB - 1]))
      $fatal(1, "generation vectors missing: run scripts/gen_model_vectors.py");

    {awvalid, wvalid, bready, arvalid, rready} = '0;
    rst_n = 0;
    repeat (5) @(posedge clk);
    rst_n <= 1;
    @(posedge clk);

    $display("[1] ID and reset values");
    axi_read(ID, v);          expect_eq("ID", v, 32'h6D475054);
    axi_read(STATUS, v);      expect_eq("STATUS after reset", v, 0);
    axi_read(START_TOKEN, v); expect_eq("START_TOKEN after reset", v, BOS);

    $display("[2] generations through registers");
    generate_and_check(BOS, 0, 0, cycles_a);
    generate_and_check(0,   3, 0, v);         // data arrives 3 cycles after address
    generate_and_check(1,   0, 0, v);
    generate_and_check(13,  1, 0, v);

    $display("[3] start written while busy");
    generate_and_check(BOS, 0, 1, cycles_b);
    expect_eq("CYCLES with an ignored restart", cycles_b, cycles_a);
    axi_read(STATUS, v);      expect_eq("STATUS when finished", v, 2);

    if (n_errors == 0) $display("PASS: AXI register interface and generations match Python");
    else               $display("FAIL: %0d errors", n_errors);
    $finish;
  end

endmodule
