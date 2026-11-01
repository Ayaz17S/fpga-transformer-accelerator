// axi_gpt.sv - AXI4-Lite register interface around the generator.
//
// This is how the Zynq's ARM cores talk to the accelerator: every register
// below is a 32-bit word the CPU reads or writes at (base address + offset).
// On PYNQ that is just mmio.write(offset, value) / mmio.read(offset).
//
//   offset  name         access  meaning
//   0x00    CTRL         W       bit 0: write 1 to start a generation (ignored while busy)
//   0x04    STATUS       R       bit 0: busy, bit 1: done (set when a generation
//                                finishes, cleared by the next start)
//   0x08    START_TOKEN  R/W     bits 4:0: token fed in at position 0 (26 = BOS)
//   0x0C    GEN_LEN      R       number of generated tokens, 0 .. 16
//   0x10    CYCLES       R       clock cycles the last generation took
//   0x14    ID           R       constant 0x6D475054 ("mGPT"): checks the overlay is loaded
//   0x40 +  TOKENS[i]    R       generated token i (i = 0 .. 15) at 0x40 + 4*i
//
// AXI4-Lite in one paragraph: a write is two handshakes, address (AW) and
// data (W), answered by a response (B); a read is an address (AR) answered by
// data (R). Each handshake completes on a clock edge where VALID (from the
// sender) and READY (from the receiver) are both high. This slave waits for
// AW and W to both be valid and accepts them together, which the AXI rules
// allow, and handles one transaction of each kind at a time.

`timescale 1ns/1ps

module axi_gpt
  import model_pkg::*;
#(
  parameter int ADDR_W = 7
) (
  input  logic              clk,
  input  logic              rst_n,

  // write address
  input  logic [ADDR_W-1:0] awaddr,
  input  logic              awvalid,
  output logic              awready,
  // write data
  input  logic [31:0]       wdata,
  input  logic [3:0]        wstrb,
  input  logic              wvalid,
  output logic              wready,
  // write response
  output logic [1:0]        bresp,
  output logic              bvalid,
  input  logic              bready,
  // read address
  input  logic [ADDR_W-1:0] araddr,
  input  logic              arvalid,
  output logic              arready,
  // read data
  output logic [31:0]       rdata,
  output logic [1:0]        rresp,
  output logic              rvalid,
  input  logic              rready
);

  localparam logic [31:0] ID_VALUE = 32'h6D47_5054;  // "mGPT"

  // word offsets (byte offset / 4)
  localparam int W_CTRL        = 'h00 >> 2;
  localparam int W_STATUS      = 'h04 >> 2;
  localparam int W_START_TOKEN = 'h08 >> 2;
  localparam int W_GEN_LEN     = 'h0C >> 2;
  localparam int W_CYCLES      = 'h10 >> 2;
  localparam int W_ID          = 'h14 >> 2;
  localparam int W_TOKENS      = 'h40 >> 2;  // 16 words from here

  // ---------------- the accelerator ----------------
  logic        gen_start, gen_busy, gen_done;
  logic [4:0]  start_token_r;
  logic [4:0]  gen_len;
  logic [3:0]  gen_addr;
  logic [4:0]  gen_token;

  generator u_gen (
    .clk        (clk),
    .rst_n      (rst_n),
    .start      (gen_start),
    .start_token(start_token_r),
    .busy       (gen_busy),
    .done       (gen_done),
    .gen_len    (gen_len),
    .gen_addr   (gen_addr),
    .gen_token  (gen_token)
  );

  // ---------------- writes ----------------
  logic do_write;
  logic [ADDR_W-3:0] wr_word;

  assign do_write = awvalid && wvalid && !bvalid;  // accept AW and W together
  assign awready  = do_write;
  assign wready   = do_write;
  assign wr_word  = awaddr[ADDR_W-1:2];
  assign bresp    = 2'b00;                         // OKAY

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      bvalid        <= 1'b0;
      gen_start     <= 1'b0;
      start_token_r <= 5'(BOS);
    end else begin
      gen_start <= 1'b0;
      if (do_write) begin
        bvalid <= 1'b1;
        if (wstrb[0]) begin
          if (wr_word == W_CTRL && wdata[0] && !gen_busy) gen_start <= 1'b1;
          if (wr_word == W_START_TOKEN)                   start_token_r <= wdata[4:0];
        end
      end else if (bvalid && bready) begin
        bvalid <= 1'b0;
      end
    end
  end

  // ---------------- status and performance counter ----------------
  logic        done_r;
  logic [31:0] cycles_r;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      done_r   <= 1'b0;
      cycles_r <= '0;
    end else begin
      if (gen_start) begin
        done_r   <= 1'b0;
        cycles_r <= '0;
      end else begin
        if (gen_busy) cycles_r <= cycles_r + 1'b1;
        if (gen_done) done_r   <= 1'b1;
      end
    end
  end

  // ---------------- reads ----------------
  logic [ADDR_W-3:0] rd_word;
  logic [31:0]       rd_value;

  assign arready  = !rvalid;                      // one read at a time
  assign rd_word  = araddr[ADDR_W-1:2];
  assign gen_addr = rd_word[3:0];                 // token index in the TOKENS window
  assign rresp    = 2'b00;                        // OKAY

  always_comb begin
    rd_value = '0;
    if (rd_word >= W_TOKENS) rd_value = 32'(gen_token);
    else case (rd_word)
      W_STATUS:      rd_value = {30'b0, done_r, gen_busy};
      W_START_TOKEN: rd_value = 32'(start_token_r);
      W_GEN_LEN:     rd_value = 32'(gen_len);
      W_CYCLES:      rd_value = cycles_r;
      W_ID:          rd_value = ID_VALUE;
      default:       rd_value = '0;
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rvalid <= 1'b0;
    end else if (arvalid && arready) begin
      rvalid <= 1'b1;
      rdata  <= rd_value;
    end else if (rvalid && rready) begin
      rvalid <= 1'b0;
    end
  end

endmodule
