// lut_rom.sv - single-port lookup table ROM, loaded from a .mem file.
//
// Synchronous read (1-cycle latency) so it maps onto block RAM.
// Used for the shared nonlinear-function tables (rsqrt, later exp).

`timescale 1ns/1ps

module lut_rom #(
  parameter int    DATA_W    = 16,
  parameter int    AW        = 10,
  parameter string INIT_FILE = "rsqrt_lut.mem"
) (
  input  logic              clk,
  input  logic [AW-1:0]     addr,
  output logic [DATA_W-1:0] data
);

  (* rom_style = "block" *) logic [DATA_W-1:0] mem [2**AW];

  initial $readmemh(INIT_FILE, mem);

  always_ff @(posedge clk) data <= mem[addr];

endmodule
