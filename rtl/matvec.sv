// matvec.sv - matrix-vector engine: y = requant(W * x).
//
// One engine serves every linear layer of the model (wq/wk/wv/wo, fc1, fc2,
// lm_head). A command gives the layer's weight base address, its shape and
// its requantize shift; the engine then walks the weights row by row with a
// single MAC.
//
// Weights are row-major W[out][in] (see contract doc), so the weight address
// simply counts up from w_base: w_base + o*n_in + i.
//
// Memory ports: w and x are read ports with 1-cycle read latency (what a
// BRAM gives); y is a write port. The memories live outside this module so
// the controller can point the engine at whichever buffers a layer needs.
//
// Timing: n_in*n_out cycles to issue, plus 6 cycles of pipeline
// (1 memory read + 3 MAC + 2 requant), counted from the cycle after start.
// y_we/y_addr/y_data come straight from registers; done pulses in the same
// cycle as the last y write.

`timescale 1ns/1ps

module matvec #(
  parameter int DATA_W = 16,
  parameter int ACC_W  = 48,
  parameter int W_AW   = 13,   // weight address bits (8192 words)
  parameter int X_AW   = 6,    // x buffer address bits (64 entries)
  parameter int Y_AW   = 6,    // y buffer address bits (64 entries)
  parameter int DIM_W  = 7,    // holds n_in / n_out up to 64
  parameter int SH_W   = 6
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // command (sampled when start is high and the engine is idle)
  input  logic                     start,
  input  logic [W_AW-1:0]          w_base,
  input  logic [DIM_W-1:0]         n_in,    // 1 .. 2^X_AW
  input  logic [DIM_W-1:0]         n_out,   // 1 .. 2^Y_AW
  input  logic [SH_W-1:0]          shift,
  output logic                     busy,
  output logic                     done,

  // weight read port
  output logic [W_AW-1:0]          w_addr,
  input  logic signed [DATA_W-1:0] w_data,

  // x read port
  output logic [X_AW-1:0]          x_addr,
  input  logic signed [DATA_W-1:0] x_data,

  // y write port
  output logic                     y_we,
  output logic [Y_AW-1:0]          y_addr,
  output logic signed [DATA_W-1:0] y_data
);

  // ---------------- command registers ----------------
  logic [W_AW-1:0]  w_ptr;
  logic [DIM_W-1:0] n_in_r, n_out_r;
  logic [SH_W-1:0]  shift_r;

  // ---------------- issue side: walk i (columns) and o (rows) ----------------
  typedef enum logic [1:0] {IDLE, ISSUE, DRAIN} state_t;
  state_t state;

  logic [DIM_W-1:0] i_cnt, o_cnt;
  logic row_last, mat_last;

  assign row_last = (i_cnt == n_in_r - 1);
  assign mat_last = row_last && (o_cnt == n_out_r - 1);

  assign busy   = (state != IDLE);
  assign w_addr = w_ptr;
  assign x_addr = i_cnt[X_AW-1:0];

  // ---------------- result side ----------------
  logic                     mac_valid;
  logic signed [ACC_W-1:0]  mac_acc;
  logic                     rq_valid;
  logic signed [DATA_W-1:0] rq_out;
  logic [DIM_W-1:0]         y_cnt;   // index of the result currently on rq_out
  logic                     final_result;

  assign final_result = rq_valid && (y_cnt == n_out_r - 1);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE:    if (start)        state <= ISSUE;
        ISSUE:   if (mat_last)     state <= DRAIN;
        DRAIN:   if (final_result) state <= IDLE;
        default:                   state <= IDLE;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (state == IDLE && start) begin
      w_ptr   <= w_base;
      n_in_r  <= n_in;
      n_out_r <= n_out;
      shift_r <= shift;
      i_cnt   <= '0;
      o_cnt   <= '0;
      y_cnt   <= '0;
    end else begin
      if (state == ISSUE) begin
        w_ptr <= w_ptr + 1'b1;
        if (row_last) begin
          i_cnt <= '0;
          o_cnt <= o_cnt + 1'b1;
        end else begin
          i_cnt <= i_cnt + 1'b1;
        end
      end
      if (rq_valid) y_cnt <= y_cnt + 1'b1;
    end
  end

  // ---------------- align flags with the 1-cycle memory read ----------------
  logic d_valid, d_first, d_last;

  always_ff @(posedge clk) begin
    if (!rst_n) d_valid <= 1'b0;
    else        d_valid <= (state == ISSUE);
    d_first <= (i_cnt == 0);
    d_last  <= row_last;
  end

  // ---------------- datapath ----------------
  mac #(.A_W(DATA_W), .B_W(DATA_W), .ACC_W(ACC_W)) u_mac (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (d_valid),
    .in_first (d_first),
    .in_last  (d_last),
    .a        (w_data),
    .b        (x_data),
    .out_valid(mac_valid),
    .acc      (mac_acc)
  );

  requant #(.IN_W(ACC_W), .OUT_W(DATA_W), .SH_W(SH_W)) u_requant (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (mac_valid),
    .in       (mac_acc),
    .shift    (shift_r),
    .out_valid(rq_valid),
    .out      (rq_out)
  );

  // requant's outputs and y_cnt are all registers, so the write port is
  // driven directly with no extra output stage.
  assign y_we   = rq_valid;
  assign y_addr = y_cnt[Y_AW-1:0];
  assign y_data = rq_out;
  assign done   = final_result;

endmodule
