// axi_gpt_top.v - plain-Verilog shell around axi_gpt for the Vivado block design.
//
// A block design can reference an RTL module directly ("module reference"),
// but the referenced top must be Verilog or VHDL; SystemVerilog is fine
// further down. So this file only renames ports. The s_axi_* names and the
// X_INTERFACE attributes let Vivado recognise an AXI4-Lite slave plus its
// clock and reset, so connection automation can wire it to the Zynq.

`timescale 1ns/1ps

module axi_gpt_top #(
  parameter C_S_AXI_ADDR_WIDTH = 7
) (
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET s_axi_aresetn" *)
  input  wire                          s_axi_aclk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
  (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input  wire                          s_axi_aresetn,

  input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
  input  wire [2:0]                    s_axi_awprot,
  input  wire                          s_axi_awvalid,
  output wire                          s_axi_awready,
  input  wire [31:0]                   s_axi_wdata,
  input  wire [3:0]                    s_axi_wstrb,
  input  wire                          s_axi_wvalid,
  output wire                          s_axi_wready,
  output wire [1:0]                    s_axi_bresp,
  output wire                          s_axi_bvalid,
  input  wire                          s_axi_bready,
  input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
  input  wire [2:0]                    s_axi_arprot,
  input  wire                          s_axi_arvalid,
  output wire                          s_axi_arready,
  output wire [31:0]                   s_axi_rdata,
  output wire [1:0]                    s_axi_rresp,
  output wire                          s_axi_rvalid,
  input  wire                          s_axi_rready
);

  axi_gpt #(.ADDR_W(C_S_AXI_ADDR_WIDTH)) u_axi_gpt (
    .clk    (s_axi_aclk),
    .rst_n  (s_axi_aresetn),
    .awaddr (s_axi_awaddr),
    .awvalid(s_axi_awvalid),
    .awready(s_axi_awready),
    .wdata  (s_axi_wdata),
    .wstrb  (s_axi_wstrb),
    .wvalid (s_axi_wvalid),
    .wready (s_axi_wready),
    .bresp  (s_axi_bresp),
    .bvalid (s_axi_bvalid),
    .bready (s_axi_bready),
    .araddr (s_axi_araddr),
    .arvalid(s_axi_arvalid),
    .arready(s_axi_arready),
    .rdata  (s_axi_rdata),
    .rresp  (s_axi_rresp),
    .rvalid (s_axi_rvalid),
    .rready (s_axi_rready)
  );

endmodule
