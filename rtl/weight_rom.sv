// weight_rom.sv - all model weights in one on-chip ROM, two read ports.
//
// The contents come from a .mem file (one 4-digit hex word per line, the
// layout in model_pkg) loaded with $readmemh. Vivado reads the same file at
// synthesis and bakes it into the BRAM's initial contents, so no DDR or
// loading logic is needed.
//
// Both ports are synchronous reads (1-cycle latency) because that is what a
// block RAM provides; an asynchronous read would force the ROM into LUTs.
// Two ports cost nothing extra: every BRAM is dual-port. The embedding unit
// uses both at once (wte row + wpe row); the matvec engine uses port A.

`timescale 1ns/1ps

module weight_rom #(
  parameter int    DATA_W    = 16,
  parameter int    AW        = 13,
  parameter int    DEPTH     = 4192,
  parameter string INIT_FILE = "weights_all.mem"
) (
  input  logic                     clk,

  input  logic [AW-1:0]            a_addr,
  output logic signed [DATA_W-1:0] a_data,

  input  logic [AW-1:0]            b_addr,
  output logic signed [DATA_W-1:0] b_data
);

  (* rom_style = "block" *) logic signed [DATA_W-1:0] mem [DEPTH];

  initial $readmemh(INIT_FILE, mem);

  always_ff @(posedge clk) begin
    a_data <= mem[a_addr];
    b_data <= mem[b_addr];
  end

endmodule
