// attention.sv - causal multi-head self-attention for one token at position t.
//
// Bit-exact twin of attention_fx() in scripts/fixedpoint.py. For each head h
// (dims 4h .. 4h+3):
//   SCORE  score_j = requant(q_h . k_h[j], 11)       j = 0..t   (4 MACs each)
//   SOFTMAX p      = softmax(score_0..t)             (softmax.sv)
//   AV     out_d   = requant(sum_j p_j v[j][d], 14)  d in head  (t+1 MACs each)
// Positions j > t are never read: that is the causal mask, for free.
//
// The KV cache lives here: 16 slots x 16 dims for k and for v. The k and v
// rows of the current token are written through the cache write port (by the
// wk / wv matvec runs) before start; earlier rows stay from earlier tokens.
// The scale 1/sqrt(HEAD_DIM) = 0.5 is one extra bit of requantize shift.
//
// One MAC serves SCORE and AV; softmax has its own small datapath.
// done pulses in the same cycle as the last output write.

`timescale 1ns/1ps

module attention
  import model_pkg::*;
#(
  parameter string EXP_LUT_FILE = "exp_lut.mem"
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // KV cache write port: row wr_slot, element wr_dim
  input  logic                     k_we,
  input  logic                     v_we,
  input  logic [3:0]               wr_slot,
  input  logic [3:0]               wr_dim,
  input  logic signed [DATA_W-1:0] wr_data,

  // command (sampled when start is high and the unit is idle)
  input  logic                     start,
  input  logic [3:0]               pos,      // t: this token's position
  output logic                     busy,
  output logic                     done,

  // q read port (1-cycle latency)
  output logic [3:0]               q_addr,
  input  logic signed [DATA_W-1:0] q_data,

  // output write port (concatenated heads, 16 values)
  output logic                     o_we,
  output logic [3:0]               o_addr,
  output logic signed [DATA_W-1:0] o_data
);

  typedef enum logic [2:0] {
    IDLE, SC_ISSUE, SC_WAIT, SM_START, SM_WAIT, AV_ISSUE, AV_WAIT
  } state_t;
  state_t state;

  logic [3:0] t_r;
  logic [1:0] h;         // head
  logic [3:0] j;         // cache position
  logic [1:0] d;         // dim within the head
  logic [4:0] s_cnt;     // scores written for this head
  logic [3:0] o_cnt;     // outputs written

  logic av_phase;
  assign av_phase = (state == AV_ISSUE) || (state == AV_WAIT);
  assign busy     = (state != IDLE);

  // ---------------- KV cache (address = {slot, dim}) ----------------
  logic signed [DATA_W-1:0] k_cache [256];
  logic signed [DATA_W-1:0] v_cache [256];
  logic signed [DATA_W-1:0] k_rd, v_rd;

  always_ff @(posedge clk) begin
    if (k_we) k_cache[{wr_slot, wr_dim}] <= wr_data;
    if (v_we) v_cache[{wr_slot, wr_dim}] <= wr_data;
    k_rd <= k_cache[{j, h, d}];
    v_rd <= v_cache[{j, h, d}];
  end

  assign q_addr = {h, d};

  // ---------------- score and probability registers ----------------
  logic signed [DATA_W-1:0] s_reg [16];
  logic signed [DATA_W-1:0] p_reg [16];
  logic signed [DATA_W-1:0] p_rd;

  // softmax ports
  logic                     sm_start, sm_busy, sm_done;
  logic [3:0]               sm_s_addr;
  logic signed [DATA_W-1:0] sm_s_data;
  logic                     sm_p_we;
  logic [3:0]               sm_p_addr;
  logic signed [DATA_W-1:0] sm_p_data;

  always_ff @(posedge clk) begin
    sm_s_data <= s_reg[sm_s_addr];               // 1-cycle read for softmax
    if (sm_p_we) p_reg[sm_p_addr] <= sm_p_data;
    p_rd <= p_reg[j];                            // 1-cycle read for AV
  end

  assign sm_start = (state == SM_START);

  softmax #(.LUT_FILE(EXP_LUT_FILE)) u_softmax (
    .clk   (clk),
    .rst_n (rst_n),
    .start (sm_start),
    .n     (5'(t_r) + 5'd1),
    .busy  (sm_busy),
    .done  (sm_done),
    .s_addr(sm_s_addr),
    .s_data(sm_s_data),
    .p_we  (sm_p_we),
    .p_addr(sm_p_addr),
    .p_data(sm_p_data)
  );

  // ---------------- MAC: q.k in SCORE, p.v in AV ----------------
  logic rd_valid, rd_first, rd_last, rd_av;
  logic                    mac_valid;
  logic signed [ACC_W-1:0] mac_acc;

  // operands arrive one cycle after their addresses, so the flags follow suit
  always_ff @(posedge clk) begin
    if (!rst_n) rd_valid <= 1'b0;
    else        rd_valid <= (state == SC_ISSUE) || (state == AV_ISSUE);
    rd_av    <= (state == AV_ISSUE);
    rd_first <= (state == AV_ISSUE) ? (j == 0)   : (d == 0);
    rd_last  <= (state == AV_ISSUE) ? (j == t_r) : (d == HEAD_DIM - 1);
  end

  mac #(.A_W(DATA_W), .B_W(DATA_W), .ACC_W(ACC_W)) u_mac (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (rd_valid),
    .in_first (rd_first),
    .in_last  (rd_last),
    .a        (rd_av ? p_rd : q_data),
    .b        (rd_av ? v_rd : k_rd),
    .out_valid(mac_valid),
    .acc      (mac_acc)
  );

  // ---------------- requantize, then route to scores or outputs ----------------
  // Every SCORE result is written before softmax starts and every AV result
  // before the next head's SCORE, so the current state says where a result goes.
  logic                     rq_valid;
  logic signed [DATA_W-1:0] rq_out;
  logic                     head_done, final_result;

  requant #(.IN_W(ACC_W), .OUT_W(DATA_W), .SH_W(6)) u_requant (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (mac_valid),
    .in       (mac_acc),
    .shift    (av_phase ? 6'(ATT_OUT_SHIFT) : 6'(ATT_SCORE_SHIFT)),
    .out_valid(rq_valid),
    .out      (rq_out)
  );

  always_ff @(posedge clk) begin
    if (rq_valid && !av_phase) s_reg[s_cnt[3:0]] <= rq_out;
  end

  assign head_done    = rq_valid && av_phase && (o_cnt[1:0] == 2'd3);
  assign final_result = head_done && (h == 2'd3);
  assign o_we   = rq_valid && av_phase;
  assign o_addr = o_cnt;
  assign o_data = rq_out;
  assign done   = final_result;

  // ---------------- control ----------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE:     if (start)                          state <= SC_ISSUE;
        SC_ISSUE: if (d == 2'd3 && j == t_r)          state <= SC_WAIT;
        SC_WAIT:  if (s_cnt == 5'(t_r) + 5'd1)        state <= SM_START;
        SM_START:                                     state <= SM_WAIT;
        SM_WAIT:  if (sm_done)                        state <= AV_ISSUE;
        AV_ISSUE: if (d == 2'd3 && j == t_r)          state <= AV_WAIT;
        AV_WAIT:  if (final_result)                   state <= IDLE;
                  else if (head_done)                 state <= SC_ISSUE;
        default:                                      state <= IDLE;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (state == IDLE && start) begin
      t_r   <= pos;
      h     <= '0;
      j     <= '0;
      d     <= '0;
      s_cnt <= '0;
      o_cnt <= '0;
    end else begin
      case (state)
        // SCORE walks j (outer) x d (inner): one 4-element dot per position
        SC_ISSUE: begin
          d <= d + 1'b1;
          if (d == 2'd3) j <= (j == t_r) ? '0 : j + 1'b1;
        end
        // AV walks d (outer) x j (inner): one (t+1)-element dot per dim
        AV_ISSUE: begin
          if (j == t_r) begin
            j <= '0;
            d <= d + 1'b1;
          end else begin
            j <= j + 1'b1;
          end
        end
        AV_WAIT: if (head_done && !final_result) begin
          h     <= h + 1'b1;
          s_cnt <= '0;
        end
        default: ;
      endcase
      if (rq_valid && !av_phase) s_cnt <= s_cnt + 1'b1;
      if (rq_valid &&  av_phase) o_cnt <= o_cnt + 1'b1;
    end
  end

endmodule
