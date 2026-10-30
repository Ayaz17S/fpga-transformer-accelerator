// generator.sv - greedy autoregressive generation around gpt_core.
//
// Bit-exact twin of generate_fx() in scripts/fixedpoint.py:
//
//   tok = start_token
//   for pos in 0..15:
//       tok = gpt_core(tok, pos).next_token
//       if tok == BOS: stop
//       out[pos] = tok
//
// "Autoregressive" just means each output becomes the next input. The KV
// cache inside gpt_core keeps k and v from every earlier position, so step
// pos only computes the new token's row and attends over rows 0..pos.
//
// Generation stops when the model predicts BOS (end of name) or after 16
// tokens, when the context is full. gen_len then holds how many tokens were
// produced, and gen_tokens[0 .. gen_len-1] can be read through gen_addr.

`timescale 1ns/1ps

module generator
  import model_pkg::*;
(
  input  logic       clk,
  input  logic       rst_n,

  // command
  input  logic       start,
  input  logic [4:0] start_token,   // usually BOS
  output logic       busy,
  output logic       done,          // pulses when gen_len and gen_tokens are valid

  // result
  output logic [4:0] gen_len,       // 0 .. 16
  input  logic [3:0] gen_addr,
  output logic [4:0] gen_token
);

  typedef enum logic [1:0] {IDLE, STEP, WAIT} state_t;
  state_t state;

  logic [4:0] tok;                  // token fed in at this position
  logic [3:0] pos;
  logic [4:0] gen_tokens [CTX];

  logic       core_start, core_done;
  logic [4:0] core_next;

  assign busy       = (state != IDLE);
  assign core_start = (state == STEP);
  assign gen_token  = gen_tokens[gen_addr];

  gpt_core u_core (
    .clk        (clk),
    .rst_n      (rst_n),
    .start      (core_start),
    .token      (tok),
    .pos        (pos),
    .busy       (),
    .done       (core_done),
    .next_token (core_next),
    .logit_addr ('0),
    .logit_data ()
  );

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
      done  <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        IDLE: if (start) begin
          tok     <= start_token;
          pos     <= '0;
          gen_len <= '0;
          state   <= STEP;
        end

        STEP: state <= WAIT;          // core_start is high for this one cycle

        WAIT: if (core_done) begin
          if (core_next == 5'(BOS)) begin
            state <= IDLE;            // end of name
            done  <= 1'b1;
          end else begin
            gen_tokens[pos] <= core_next;
            gen_len         <= gen_len + 1'b1;
            if (pos == 4'(CTX - 1)) begin
              state <= IDLE;          // context full
              done  <= 1'b1;
            end else begin
              tok   <= core_next;     // feed the prediction back in
              pos   <= pos + 1'b1;
              state <= STEP;
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
