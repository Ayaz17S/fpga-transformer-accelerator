// embed.sv - token + position embedding: x[d] = wte[token][d] + wpe[pos][d].
//
// Reads one wte word and one wpe word per cycle through the ROM's two ports,
// adds them, and converts the sum from the weight format to the residual
// stream format with the shared requantize unit (contract rule 5: a format
// change without a multiply uses the same requantize function).
//
// Timing: D_MODEL cycles to issue, plus 3 cycles of pipeline
// (1 ROM read + 2 requant), counted from the cycle after start.
// done pulses in the same cycle as the last x write.

`timescale 1ns/1ps

module embed
  import model_pkg::*;
#(
  parameter int SHIFT = EMB_SHIFT
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // command (sampled when start is high and the unit is idle)
  input  logic                     start,
  input  logic [4:0]               token,   // 0 .. VOCAB-1
  input  logic [3:0]               pos,     // 0 .. CTX-1
  output logic                     busy,
  output logic                     done,

  // weight ROM read ports
  output logic [W_AW-1:0]          wa_addr, // wte row
  input  logic signed [DATA_W-1:0] wa_data,
  output logic [W_AW-1:0]          wb_addr, // wpe row
  input  logic signed [DATA_W-1:0] wb_data,

  // x write port
  output logic                     x_we,
  output logic [3:0]               x_addr,
  output logic signed [DATA_W-1:0] x_data
);

  typedef enum logic [1:0] {IDLE, ISSUE, DRAIN} state_t;
  state_t state;

  logic [W_AW-1:0] tok_row, pos_row;  // row start addresses, latched at start
  logic [3:0]      d_cnt;             // element being read
  logic [3:0]      x_cnt;             // element being written
  logic            rq_valid;
  logic            final_result;

  assign final_result = rq_valid && (x_cnt == D_MODEL - 1);

  assign busy    = (state != IDLE);
  assign wa_addr = tok_row + d_cnt;
  assign wb_addr = pos_row + d_cnt;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE:    if (start)                   state <= ISSUE;
        ISSUE:   if (d_cnt == D_MODEL - 1)    state <= DRAIN;
        DRAIN:   if (final_result)            state <= IDLE;
        default:                              state <= IDLE;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (state == IDLE && start) begin
      tok_row <= W_AW'(WTE_BASE) + W_AW'(token) * D_MODEL;
      pos_row <= W_AW'(WPE_BASE) + W_AW'(pos)   * D_MODEL;
      d_cnt   <= '0;
      x_cnt   <= '0;
    end else begin
      if (state == ISSUE) d_cnt <= d_cnt + 1'b1;
      if (rq_valid)       x_cnt <= x_cnt + 1'b1;
    end
  end

  // ROM data arrives one cycle after the address
  logic rd_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) rd_valid <= 1'b0;
    else        rd_valid <= (state == ISSUE);
  end

  requant #(.IN_W(ACC_W), .OUT_W(DATA_W), .SH_W(6)) u_requant (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (rd_valid),
    .in       (ACC_W'(wa_data) + ACC_W'(wb_data)),  // sign-extend, then add
    .shift    (6'(SHIFT)),
    .out_valid(rq_valid),
    .out      (x_data)
  );

  assign x_we   = rq_valid;
  assign x_addr = x_cnt;
  assign done   = final_result;

endmodule
