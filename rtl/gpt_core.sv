// gpt_core.sv - the whole model: one token in, logits and next token out.
//
// Bit-exact twin of GptFx.step() in scripts/fixedpoint.py. A start command
// runs one token step through twelve operations, in order:
//
//   step     unit       reads    writes        notes
//   EMBED    embed      ROM      E             wte[token] + wpe[pos]
//   NORM0    rmsnorm    E        X             Q6.10, starts the residual stream
//   NORM1    rmsnorm    X        H             Q4.12
//   WQ       matvec     H        Q
//   WK       matvec     H        K cache[pos]
//   WV       matvec     H        V cache[pos]
//   ATTN     attention  Q, KV    A
//   WO       matvec     A        X += y        residual add fused into the write
//   NORM2    rmsnorm    X        H
//   FC1      matvec     H        F = relu(y)   ReLU fused into the write
//   FC2      matvec     F        X += y        residual add fused into the write
//   LM       matvec     X        L             argmax tracked as logits are written
//
// The controller starts one unit, waits for its done, and moves to the next
// step. Only one unit is busy at a time, so the weight ROM port and the
// buffer read ports are simple muxes on the current step.

`timescale 1ns/1ps

module gpt_core
  import model_pkg::*;
#(
  parameter string WEIGHT_FILE = "weights_all.mem",
  parameter string RSQRT_FILE  = "rsqrt_lut.mem",
  parameter string EXP_FILE    = "exp_lut.mem"
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // command (sampled when start is high and the core is idle)
  input  logic                     start,
  input  logic [4:0]               token,
  input  logic [3:0]               pos,
  output logic                     busy,
  output logic                     done,        // pulses when next_token is valid
  output logic [4:0]               next_token,

  // logit read port (combinational), valid after done
  input  logic [4:0]               logit_addr,
  output logic signed [DATA_W-1:0] logit_data
);

  // ---------------- activation buffers ----------------
  logic signed [DATA_W-1:0] e_buf [D_MODEL];  // embedding
  logic signed [DATA_W-1:0] x_buf [D_MODEL];  // residual stream
  logic signed [DATA_W-1:0] h_buf [D_MODEL];  // rmsnorm output
  logic signed [DATA_W-1:0] q_buf [D_MODEL];  // query
  logic signed [DATA_W-1:0] a_buf [D_MODEL];  // attention output
  logic signed [DATA_W-1:0] f_buf [D_FF];     // MLP hidden, after ReLU
  logic signed [DATA_W-1:0] l_buf [VOCAB];    // logits

  assign logit_data = l_buf[logit_addr];

  // ---------------- controller ----------------
  typedef enum logic [3:0] {
    S_EMBED, S_NORM0, S_NORM1, S_WQ, S_WK, S_WV, S_ATTN,
    S_WO, S_NORM2, S_FC1, S_FC2, S_LM
  } step_t;

  typedef enum logic [1:0] {IDLE, ISSUE, WAIT} state_t;

  state_t     state;
  step_t      step;
  logic [4:0] token_r;
  logic [3:0] pos_r;
  logic       unit_done;

  assign busy = (state != IDLE);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
      done  <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        IDLE: if (start) begin
          token_r <= token;
          pos_r   <= pos;
          step    <= S_EMBED;
          state   <= ISSUE;
        end
        ISSUE: state <= WAIT;
        WAIT: if (unit_done) begin
          if (step == S_LM) begin
            state <= IDLE;
            done  <= 1'b1;
          end else begin
            step  <= step_t'(step + 1'b1);
            state <= ISSUE;
          end
        end
        default: state <= IDLE;
      endcase
    end
  end

  // ---------------- per-step decode ----------------
  typedef enum logic [1:0] {SRC_H, SRC_A, SRC_F, SRC_X} mv_src_t;
  typedef enum logic [2:0] {DST_Q, DST_K, DST_V, DST_XRES, DST_FRELU, DST_LOGIT} mv_dst_t;

  logic            is_mv, is_rms;
  logic [W_AW-1:0] mv_base;
  logic [6:0]      mv_n_in, mv_n_out;
  logic [5:0]      mv_shift;
  mv_src_t         mv_src;
  mv_dst_t         mv_dst;
  logic [3:0]      rms_frac;

  always_comb begin
    is_mv    = 1'b1;
    is_rms   = 1'b0;
    mv_base  = '0;
    mv_n_in  = 7'(D_MODEL);
    mv_n_out = 7'(D_MODEL);
    mv_shift = 6'(SHIFT_FROM_NORM);
    mv_src   = SRC_H;
    mv_dst   = DST_Q;
    rms_frac = 4'(NORM_FRAC);
    case (step)
      S_EMBED: is_mv = 1'b0;
      S_NORM0: begin is_mv = 1'b0; is_rms = 1'b1; rms_frac = 4'(NORM0_FRAC); end
      S_NORM1,
      S_NORM2: begin is_mv = 1'b0; is_rms = 1'b1; end
      S_ATTN:  is_mv = 1'b0;
      S_WQ:    begin mv_base = W_AW'(WQ_BASE); mv_dst = DST_Q; end
      S_WK:    begin mv_base = W_AW'(WK_BASE); mv_dst = DST_K; end
      S_WV:    begin mv_base = W_AW'(WV_BASE); mv_dst = DST_V; end
      S_WO:    begin mv_base = W_AW'(WO_BASE); mv_src = SRC_A; mv_dst = DST_XRES;
                     mv_shift = 6'(SHIFT_FROM_ACT); end
      S_FC1:   begin mv_base = W_AW'(FC1_BASE); mv_n_out = 7'(D_FF); mv_dst = DST_FRELU; end
      S_FC2:   begin mv_base = W_AW'(FC2_BASE); mv_n_in = 7'(D_FF); mv_src = SRC_F;
                     mv_dst = DST_XRES; mv_shift = 6'(SHIFT_FROM_ACT); end
      S_LM:    begin mv_base = W_AW'(LM_BASE); mv_n_out = 7'(VOCAB); mv_src = SRC_X;
                     mv_dst = DST_LOGIT; mv_shift = 6'(SHIFT_FROM_ACT); end
      default: ;
    endcase
  end

  // ---------------- weight ROM (shared by embed and matvec) ----------------
  logic [W_AW-1:0]          rom_a_addr, rom_b_addr;
  logic signed [DATA_W-1:0] rom_a_data, rom_b_data;

  weight_rom #(.DATA_W(DATA_W), .AW(W_AW), .DEPTH(N_WEIGHTS),
               .INIT_FILE(WEIGHT_FILE)) u_rom (
    .clk   (clk),
    .a_addr(rom_a_addr), .a_data(rom_a_data),
    .b_addr(rom_b_addr), .b_data(rom_b_data)
  );

  // ---------------- embed ----------------
  logic                     emb_done;
  logic [W_AW-1:0]          emb_wa_addr, emb_wb_addr;
  logic                     emb_we;
  logic [3:0]               emb_addr;
  logic signed [DATA_W-1:0] emb_data;

  embed u_embed (
    .clk(clk), .rst_n(rst_n),
    .start  (state == ISSUE && step == S_EMBED),
    .token  (token_r),
    .pos    (pos_r),
    .busy   (),
    .done   (emb_done),
    .wa_addr(emb_wa_addr), .wa_data(rom_a_data),
    .wb_addr(emb_wb_addr), .wb_data(rom_b_data),
    .x_we   (emb_we), .x_addr(emb_addr), .x_data(emb_data)
  );

  // ---------------- rmsnorm ----------------
  logic                     rms_done;
  logic [3:0]               rms_x_addr;
  logic signed [DATA_W-1:0] rms_x_data;
  logic                     rms_we;
  logic [3:0]               rms_addr;
  logic signed [DATA_W-1:0] rms_data;

  rmsnorm #(.LUT_FILE(RSQRT_FILE)) u_rmsnorm (
    .clk(clk), .rst_n(rst_n),
    .start   (state == ISSUE && is_rms),
    .out_frac(rms_frac),
    .busy    (),
    .done    (rms_done),
    .x_addr  (rms_x_addr), .x_data(rms_x_data),
    .h_we    (rms_we), .h_addr(rms_addr), .h_data(rms_data)
  );

  // ---------------- matvec ----------------
  logic                     mv_done;
  logic [W_AW-1:0]          mv_w_addr;
  logic [5:0]               mv_x_addr;
  logic signed [DATA_W-1:0] mv_x_data;
  logic                     mv_we;
  logic [5:0]               mv_y_addr;
  logic signed [DATA_W-1:0] mv_y_data;

  matvec #(.DATA_W(DATA_W), .ACC_W(ACC_W), .W_AW(W_AW)) u_matvec (
    .clk(clk), .rst_n(rst_n),
    .start (state == ISSUE && is_mv),
    .w_base(mv_base),
    .n_in  (mv_n_in),
    .n_out (mv_n_out),
    .shift (mv_shift),
    .busy  (),
    .done  (mv_done),
    .w_addr(mv_w_addr), .w_data(rom_a_data),
    .x_addr(mv_x_addr), .x_data(mv_x_data),
    .y_we  (mv_we), .y_addr(mv_y_addr), .y_data(mv_y_data)
  );

  // ---------------- attention (owns the KV cache) ----------------
  logic                     att_done;
  logic [3:0]               att_q_addr;
  logic signed [DATA_W-1:0] att_q_data;
  logic                     att_we;
  logic [3:0]               att_addr;
  logic signed [DATA_W-1:0] att_data;

  attention #(.EXP_LUT_FILE(EXP_FILE)) u_attention (
    .clk(clk), .rst_n(rst_n),
    .k_we   (mv_we && mv_dst == DST_K),
    .v_we   (mv_we && mv_dst == DST_V),
    .wr_slot(pos_r),
    .wr_dim (mv_y_addr[3:0]),
    .wr_data(mv_y_data),
    .start  (state == ISSUE && step == S_ATTN),
    .pos    (pos_r),
    .busy   (),
    .done   (att_done),
    .q_addr (att_q_addr), .q_data(att_q_data),
    .o_we   (att_we), .o_addr(att_addr), .o_data(att_data)
  );

  always_comb begin
    case (step)
      S_EMBED: unit_done = emb_done;
      S_ATTN:  unit_done = att_done;
      default: unit_done = is_rms ? rms_done : mv_done;
    endcase
  end

  // ---------------- ROM and buffer read muxes ----------------
  assign rom_a_addr = (step == S_EMBED) ? emb_wa_addr : mv_w_addr;
  assign rom_b_addr = emb_wb_addr;

  // every unit expects a 1-cycle registered read, like a BRAM
  always_ff @(posedge clk) begin
    rms_x_data <= (step == S_NORM0) ? e_buf[rms_x_addr] : x_buf[rms_x_addr];
    att_q_data <= q_buf[att_q_addr];
    case (mv_src)
      SRC_H:   mv_x_data <= h_buf[mv_x_addr[3:0]];
      SRC_A:   mv_x_data <= a_buf[mv_x_addr[3:0]];
      SRC_F:   mv_x_data <= f_buf[mv_x_addr];
      default: mv_x_data <= x_buf[mv_x_addr[3:0]];
    endcase
  end

  // ---------------- buffer writes ----------------
  function automatic logic signed [DATA_W-1:0] sat_add(logic signed [DATA_W-1:0] a,
                                                       logic signed [DATA_W-1:0] b);
    logic signed [DATA_W:0] s;
    s = a + b;  // 17-bit context: no overflow before the clamp
    if      (s >  (2**(DATA_W-1) - 1)) return DATA_W'(2**(DATA_W-1) - 1);
    else if (s < -(2**(DATA_W-1)))     return DATA_W'(-(2**(DATA_W-1)));
    else                               return s[DATA_W-1:0];
  endfunction

  logic signed [DATA_W-1:0] best_logit;

  always_ff @(posedge clk) begin
    if (emb_we) e_buf[emb_addr] <= emb_data;
    if (rms_we) begin
      if (step == S_NORM0) x_buf[rms_addr] <= rms_data;
      else                 h_buf[rms_addr] <= rms_data;
    end
    if (att_we) a_buf[att_addr] <= att_data;

    if (mv_we) begin
      case (mv_dst)
        DST_Q:     q_buf[mv_y_addr[3:0]] <= mv_y_data;
        DST_XRES:  x_buf[mv_y_addr[3:0]] <= sat_add(x_buf[mv_y_addr[3:0]], mv_y_data);
        DST_FRELU: f_buf[mv_y_addr]      <= (mv_y_data < 0) ? '0 : mv_y_data;
        DST_LOGIT: begin
          l_buf[mv_y_addr] <= mv_y_data;
          // logits arrive in index order; a strict > keeps the lowest index on ties
          if (mv_y_addr == 0 || mv_y_data > best_logit) begin
            best_logit <= mv_y_data;
            next_token <= mv_y_addr[4:0];
          end
        end
        default: ;  // K and V go straight into the attention unit's cache
      endcase
    end
  end

endmodule
