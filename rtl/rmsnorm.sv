// rmsnorm.sv - h = x / sqrt(mean(x^2) + eps), no learned gain.
//
// Bit-exact twin of rmsnorm_fx() in scripts/fixedpoint.py:
//   M     = sum(x_i^2) + D_MODEL*EPS      (the /D is folded into the shift)
//   e     = even bit position at/below M's leading one, so M/2^e is in [1,4)
//   idx   = M scaled so its top LUT_BITS bits index the table
//   G     = RSQRT_LUT[idx]                (~ 2^15 / sqrt(M/2^e))
//   h_i   = requant(x_i * G, SHIFT_BASE + e/2 - out_frac)
//
// Three phases share one MAC (one DSP48):
//   SQ    stream x through the MAC as x*x -> sum of squares
//   NORM  M, e, idx, table read            (a few cycles, no multiplier)
//   SCALE stream x through the MAC again as x*G, one product per element,
//         requantize, write h
// out_frac picks the output format: 12 -> Q4.12, 10 -> Q6.10.
// done pulses in the same cycle as the last h write.

`timescale 1ns/1ps

module rmsnorm
  import model_pkg::*;
#(
  parameter string LUT_FILE = "rsqrt_lut.mem"
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // command (sampled when start is high and the unit is idle)
  input  logic                     start,
  input  logic [3:0]               out_frac,
  output logic                     busy,
  output logic                     done,

  // x read port (1-cycle latency)
  output logic [3:0]               x_addr,
  input  logic signed [DATA_W-1:0] x_data,

  // h write port
  output logic                     h_we,
  output logic [3:0]               h_addr,
  output logic signed [DATA_W-1:0] h_data
);

  localparam int FB  = RMS_LUT_BITS - 2;  // fraction bits of M/2^e in the index
  localparam int M_W = 36;                // sum of 16 squares of 16-bit values + eps
  localparam int E_W = 6;                 // bit position within M

  typedef enum logic [2:0] {
    IDLE, SQ_ISSUE, SQ_WAIT, NORM_E, NORM_IDX, NORM_LUT, SC_ISSUE, SC_DRAIN
  } state_t;
  state_t state;

  logic [3:0]  i_cnt;      // element being read
  logic [3:0]  h_cnt;      // element being written
  logic [3:0]  out_frac_r;
  logic last_elem;

  assign last_elem = (i_cnt == D_MODEL - 1);
  assign busy      = (state != IDLE);
  assign x_addr    = i_cnt;

  // ---------------- MAC (shared by the SQ and SCALE phases) ----------------
  logic                     rd_valid, rd_first, rd_last, rd_square;
  logic [DATA_W-1:0]        lut_data;  // G: stays valid for the whole SCALE phase
  logic                     mac_valid;
  logic signed [ACC_W-1:0]  mac_acc;

  // x arrives one cycle after its address, so the flags are delayed to match
  always_ff @(posedge clk) begin
    if (!rst_n) rd_valid <= 1'b0;
    else        rd_valid <= (state == SQ_ISSUE) || (state == SC_ISSUE);
    rd_square <= (state == SQ_ISSUE);
    rd_first  <= (state == SC_ISSUE) || (i_cnt == 0);  // SCALE: every product is its own "dot"
    rd_last   <= (state == SC_ISSUE) || last_elem;
  end

  mac #(.A_W(DATA_W), .B_W(DATA_W), .ACC_W(ACC_W)) u_mac (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (rd_valid),
    .in_first (rd_first),
    .in_last  (rd_last),
    .a        (x_data),
    .b        (rd_square ? x_data : $signed(lut_data)),  // G < 2^15, so positive
    .out_valid(mac_valid),
    .acc      (mac_acc)
  );

  // ---------------- NORM phase: M -> e -> idx -> G ----------------
  logic [M_W-1:0]          m_r;
  logic [E_W-1:0]          e_r;
  logic [RMS_LUT_BITS-1:0] idx_r;
  logic [5:0]              shift_r;

  // highest set bit of M (M >= D_MODEL*EPS > 0, so there always is one)
  function automatic logic [E_W-1:0] lead_one(logic [M_W-1:0] v);
    lead_one = '0;
    for (int k = 0; k < M_W; k++)
      if (v[k]) lead_one = E_W'(k);
  endfunction

  lut_rom #(.DATA_W(DATA_W), .AW(RMS_LUT_BITS), .INIT_FILE(LUT_FILE)) u_lut (
    .clk (clk),
    .addr(idx_r),
    .data(lut_data)
  );

  always_ff @(posedge clk) begin
    case (state)
      SQ_WAIT:  if (mac_valid) m_r <= M_W'(mac_acc) + M_W'(D_MODEL * RMS_EPS_RAW);
      NORM_E:   e_r <= lead_one(m_r) & ~E_W'(1);                     // round down to even
      NORM_IDX: idx_r <= (e_r >= FB) ? RMS_LUT_BITS'(m_r >> (e_r - FB))
                                     : RMS_LUT_BITS'(m_r << (FB - e_r));
      // the table reads idx_r during NORM_LUT; G is on lut_data from the
      // first SC_ISSUE cycle, one cycle before the first x*G product needs it
      NORM_LUT: shift_r <= 6'(RMS_SHIFT_BASE + (e_r >> 1) - out_frac_r);
      default: ;
    endcase
  end

  // ---------------- requantize and write h ----------------
  logic rq_valid;
  logic final_result;

  requant #(.IN_W(ACC_W), .OUT_W(DATA_W), .SH_W(6)) u_requant (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (mac_valid && (state == SC_ISSUE || state == SC_DRAIN)),
    .in       (mac_acc),
    .shift    (shift_r),
    .out_valid(rq_valid),
    .out      (h_data)
  );

  assign final_result = rq_valid && (h_cnt == D_MODEL - 1);
  assign h_we   = rq_valid;
  assign h_addr = h_cnt;
  assign done   = final_result;

  // ---------------- control ----------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE:     if (start)        state <= SQ_ISSUE;
        SQ_ISSUE: if (last_elem)    state <= SQ_WAIT;
        SQ_WAIT:  if (mac_valid)    state <= NORM_E;
        NORM_E:                     state <= NORM_IDX;
        NORM_IDX:                   state <= NORM_LUT;
        NORM_LUT:                   state <= SC_ISSUE;
        SC_ISSUE: if (last_elem)    state <= SC_DRAIN;
        SC_DRAIN: if (final_result) state <= IDLE;
        default:                    state <= IDLE;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (state == IDLE && start) begin
      out_frac_r <= out_frac;
      i_cnt      <= '0;
      h_cnt      <= '0;
    end else begin
      if (state == SQ_ISSUE || state == SC_ISSUE) i_cnt <= i_cnt + 1'b1;  // wraps 15 -> 0
      if (rq_valid) h_cnt <= h_cnt + 1'b1;
    end
  end

endmodule
