# Out-of-context synthesis of one RTL module, with a 100 MHz clock, to check
# resources and estimated timing. Usage (from the project root):
#   vivado -mode batch -source syn/synth.tcl -tclargs <top> [part]
# Default part is the PYNQ-Z2's Zynq. Until Zynq-7000 device support is
# installed, pass an Artix-7 part (same 7-series fabric and DSP48E1), e.g.
#   -tclargs matvec xc7a35tcpg236-1
if {$argc < 1} { error "usage: -tclargs <top> \[part\]" }
set top  [lindex $argv 0]
set part xc7z020clg400-1
if {$argc > 1} { set part [lindex $argv 1] }

# Packages first, so modules that import them compile.
read_verilog -sv [glob rtl/*_pkg.sv]
read_verilog -sv [lsearch -all -inline -not [glob rtl/*.sv] *_pkg.sv]
# ROM images, so $readmemh("weights_all.mem") finds them at synthesis.
read_mem [glob data/*.mem]
synth_design -top $top -part $part -mode out_of_context
create_clock -name clk -period 10.0 [get_ports clk]

file mkdir build
report_utilization    -file build/${top}_util.rpt
report_timing_summary -file build/${top}_timing.rpt
report_utilization
report_timing_summary -max_paths 1
