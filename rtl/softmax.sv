// softmax.sv - p = softmax(s[0..n-1]) for 1..16 scores.
//
// Bit-exact twin of softmax_fx() in scripts/fixedpoint.py:
//   m   = max s
//   e_j = EXP_LUT[min(1023, (m - s_j + 8) >> 4)]   exp(-k/64), Q2.14
//   S   = sum e_j                                   in [1.0, 16.0]
//   R   = floor(2^30 / S)                           sequential divider
//   p_j = requant(e_j * R, 16)                      Q2.14
//
// Four passes over the scores:
//   MAX   read s, keep the running max                    n cycles
//   EXP   read s, look up e_j, store it, add it to S       n cycles
//   DIV   restoring division, one quotient bit per cycle   31 cycles
//   NORM  multiply each stored e_j by R, requantize, write p
// Subtracting the max first means every exp() argument is <= 0, so e_j <= 1
// and the sum can never overflow; that is why softmax needs the MAX pass.
// done pulses in the same cycle as the last p write.

`timescale 1ns/1ps

module softmax
  import model_pkg::*;
#(
  parameter string LUT_FILE = "exp_lut.mem"
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // command (sampled when start is high and the unit is idle)
  input  logic                     start,
  input  logic [4:0]               n,       // 1 .. 16
  output logic                     busy,
  output logic                     done,

  // score read port (1-cycle latency)
  output logic [3:0]               s_addr,
  input  logic signed [DATA_W-1:0] s_data,

  // probability write port
  output logic                     p_we,
  output logic [3:0]               p_addr,
  output logic signed [DATA_W-1:0] p_data
);

  localparam int SUM_W = 19;                // 16 * 2^14 needs 19 bits
  localparam int Q_W   = SM_RECIP_BITS + 1; // quotient bits produced by the divider

  typedef enum logic [2:0] {
    IDLE, MAX_ISSUE, MAX_WAIT, EXP_ISSUE, EXP_WAIT, DIV, NORM_ISSUE, NORM_DRAIN
  } state_t;
  state_t state;

  logic [4:0] n_r;
  logic [3:0] j_cnt;        // score being read / probability being issued
  logic       last_j;

  assign last_j = (j_cnt == n_r - 1);
  assign busy   = (state != IDLE);
  assign s_addr = j_cnt;

  // ---------------- score read pipeline (1-cycle latency) ----------------
  logic       rd_valid, rd_first, rd_last;
  logic [3:0] rd_j;

  always_ff @(posedge clk) begin
    if (!rst_n) rd_valid <= 1'b0;
    else        rd_valid <= (state == MAX_ISSUE) || (state == EXP_ISSUE);
    rd_first <= (j_cnt == 0);
    rd_last  <= last_j;
    rd_j     <= j_cnt;
  end

  // ---------------- MAX pass ----------------
  logic signed [DATA_W-1:0] m_r;

  always_ff @(posedge clk) begin
    if (state == MAX_WAIT || state == MAX_ISSUE) begin
      if (rd_valid && (rd_first || s_data > m_r)) m_r <= s_data;
    end
  end

  // ---------------- EXP pass: index -> table -> store and sum ----------------
  logic [EXP_LUT_BITS-1:0] idx_r;
  logic                    idx_valid, idx_last;
  logic [3:0]              idx_j;
  logic [DATA_W-1:0]       exp_data;
  logic                    e_valid, e_last;
  logic [3:0]              e_j;
  logic [DATA_W-1:0]       e_mem [16];
  logic [SUM_W-1:0]        sum_r;

  // m - s + half step, then shift. m - s can reach 65535, so it is computed in
  // 17 bits; m >= s, so the result is a non-negative number.
  logic [DATA_W:0] diff, diff_rnd, idx_full;
  assign diff     = {m_r[DATA_W-1], m_r} - {s_data[DATA_W-1], s_data};
  assign diff_rnd = diff + (1 << (EXP_IDX_SHIFT - 1));
  assign idx_full = diff_rnd >> EXP_IDX_SHIFT;

  always_ff @(posedge clk) begin
    // stage 1: table index from the score that just arrived
    if (!rst_n) idx_valid <= 1'b0;
    else        idx_valid <= rd_valid && (state == EXP_ISSUE || state == EXP_WAIT);
    idx_r    <= (idx_full > (2**EXP_LUT_BITS - 1)) ? '1 : idx_full[EXP_LUT_BITS-1:0];
    idx_last <= rd_last;
    idx_j    <= rd_j;

    // stage 2: table output is valid (1-cycle ROM read)
    if (!rst_n) e_valid <= 1'b0;
    else        e_valid <= idx_valid;
    e_last <= idx_last;
    e_j    <= idx_j;

    // stage 3: store e_j and add it to the sum
    if (e_valid) begin
      e_mem[e_j] <= exp_data;
      sum_r      <= (e_j == 0 ? '0 : sum_r) + SUM_W'(exp_data);
    end
  end

  lut_rom #(.DATA_W(DATA_W), .AW(EXP_LUT_BITS), .INIT_FILE(LUT_FILE)) u_exp (
    .clk (clk),
    .addr(idx_r),
    .data(exp_data)
  );

  // ---------------- DIV: R = floor(2^30 / sum), restoring division ----------------
  // The dividend 2^30 has a single 1 bit, so the remainder simply shifts in a
  // 1 at that step and 0s everywhere else.
  logic [SUM_W:0]   rem_r;
  logic [Q_W-1:0]   quo_r;
  logic [4:0]       bit_cnt;   // counts Q_W-1 down to 0
  logic [SUM_W:0]   rem_shift;

  assign rem_shift = {rem_r[SUM_W-1:0], (bit_cnt == SM_RECIP_BITS)};

  always_ff @(posedge clk) begin
    if (state == EXP_WAIT && e_valid && e_last) begin
      rem_r   <= '0;
      quo_r   <= '0;
      bit_cnt <= 5'(SM_RECIP_BITS);
    end else if (state == DIV) begin
      if (rem_shift >= {1'b0, sum_r}) begin
        rem_r <= rem_shift - {1'b0, sum_r};
        quo_r <= {quo_r[Q_W-2:0], 1'b1};
      end else begin
        rem_r <= rem_shift;
        quo_r <= {quo_r[Q_W-2:0], 1'b0};
      end
      bit_cnt <= bit_cnt - 1'b1;
    end
  end

  // ---------------- NORM pass: p_j = requant(e_j * R, 16) ----------------
  // R <= 2^16 needs 17 bits, so this multiplier is separate from the 16x16 MAC.
  logic                   mul_valid;
  logic signed [ACC_W-1:0] prod_r;
  logic                   rq_valid;
  logic [3:0]             p_cnt;
  logic                   final_result;

  always_ff @(posedge clk) begin
    if (!rst_n) mul_valid <= 1'b0;
    else        mul_valid <= (state == NORM_ISSUE);
    // assigning to the 48-bit prod_r makes the multiply 48 bits wide
    prod_r <= $signed({1'b0, e_mem[j_cnt]}) * $signed({1'b0, quo_r[16:0]});
  end

  requant #(.IN_W(ACC_W), .OUT_W(DATA_W), .SH_W(6)) u_requant (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (mul_valid),
    .in       (prod_r),
    .shift    (6'(SM_PROB_SHIFT)),
    .out_valid(rq_valid),
    .out      (p_data)
  );

  assign final_result = rq_valid && (p_cnt == n_r - 1);
  assign p_we   = rq_valid;
  assign p_addr = p_cnt;
  assign done   = final_result;

  // ---------------- control ----------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE:       if (start)                      state <= MAX_ISSUE;
        MAX_ISSUE:  if (last_j)                     state <= MAX_WAIT;
        MAX_WAIT:   if (rd_valid && rd_last)        state <= EXP_ISSUE;
        EXP_ISSUE:  if (last_j)                     state <= EXP_WAIT;
        EXP_WAIT:   if (e_valid && e_last)          state <= DIV;
        DIV:        if (bit_cnt == 0)               state <= NORM_ISSUE;
        NORM_ISSUE: if (last_j)                     state <= NORM_DRAIN;
        NORM_DRAIN: if (final_result)               state <= IDLE;
        default:                                    state <= IDLE;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (state == IDLE && start) begin
      n_r   <= n;
      j_cnt <= '0;
      p_cnt <= '0;
    end else begin
      if (state == MAX_ISSUE || state == EXP_ISSUE || state == NORM_ISSUE)
        j_cnt <= last_j ? '0 : j_cnt + 1'b1;
      if (rq_valid) p_cnt <= p_cnt + 1'b1;
    end
  end

endmodule
