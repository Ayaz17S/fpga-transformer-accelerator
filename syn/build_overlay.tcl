# Build the PYNQ overlay: Zynq block design + accelerator -> bitstream.
# Usage (from the project root; takes ~10-20 minutes):
#   vivado -mode batch -source syn/build_overlay.tcl
#
# Output in overlay/:
#   gpt_overlay.bit   the bitstream
#   gpt_overlay.hwh   hardware handoff: PYNQ reads it to find the IP and its
#                     address; it must sit next to the .bit with the same name
#   timing.rpt, utilization.rpt
#
# The block design is:
#   ARM (processing_system7) --M_AXI_GP0--> AXI interconnect --> axi_gpt_top
#   FCLK_CLK0 (100 MHz) clocks the interconnect and the accelerator.
#
# Synthesis and implementation run inside this Vivado process (synth_design,
# place_design, ...) instead of launch_runs: on this machine Vivado cannot
# spawn its run scripts in batch mode.

set root     [file normalize [file join [file dirname [info script]] ..]]
set part     xc7z020clg400-1
set proj_dir [file join $root build overlay_proj]
set out_dir  [file join $root overlay]
file mkdir $out_dir

create_project overlay_proj $proj_dir -part $part -force

# ---------------- sources ----------------
add_files [glob [file join $root rtl *.sv]]
add_files [file join $root rtl axi_gpt_top.v]
# ROM images, baked into block RAM at synthesis
add_files [file join $root data weights_all.mem] \
          [file join $root data rsqrt_lut.mem] \
          [file join $root data exp_lut.mem]
update_compile_order -fileset sources_1

# ---------------- block design ----------------
create_bd_design gpt_overlay

set ps7 [create_bd_cell -type ip -vlnv [get_ipdefs -filter {NAME == processing_system7}] ps7]
# PYNQ-Z2: 512 MB DDR3 (MT41K256M16 RE-125, 16-bit). The PYNQ image already
# set up DDR at boot, so for an overlay these mostly document the board.
set_property -dict [list \
  CONFIG.PCW_USE_M_AXI_GP0             {1} \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ  {100} \
  CONFIG.PCW_UIPARAM_DDR_PARTNO        {MT41K256M16 RE-125} \
  CONFIG.PCW_UIPARAM_DDR_BUS_WIDTH     {16 Bit} \
] $ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable"} $ps7

# the accelerator, referenced straight from rtl/axi_gpt_top.v
create_bd_cell -type module -reference axi_gpt_top gpt

# connect ARM -> interconnect -> accelerator, plus clock and reset
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
  Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
  Master {/ps7/M_AXI_GP0} Slave {/gpt/s_axi} \
  intc_ip {New AXI Interconnect} master_apm {0}} [get_bd_intf_pins gpt/s_axi]

assign_bd_address
validate_bd_design
save_bd_design

foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data]] {
  puts "ADDRESS: [get_property NAME $seg] at [get_property OFFSET $seg] range [get_property RANGE $seg]"
}

# generate the block design's HDL, with all IP synthesized together with the
# top (no separate out-of-context runs), and an HDL wrapper as the top
set bd_file [get_files gpt_overlay.bd]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
make_wrapper -files $bd_file -top -import
set_property top gpt_overlay_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---------------- synthesis and implementation ----------------
synth_design -top gpt_overlay_wrapper -part $part
opt_design
place_design
phys_opt_design
route_design

report_timing_summary -file [file join $out_dir timing.rpt]
report_utilization    -file [file join $out_dir utilization.rpt]
write_bitstream -force [file join $out_dir gpt_overlay.bit]

set hwh [glob -nocomplain [file join $proj_dir overlay_proj.gen sources_1 bd gpt_overlay hw_handoff gpt_overlay.hwh]]
if {$hwh eq ""} { error "gpt_overlay.hwh not found" }
file copy -force $hwh [file join $out_dir gpt_overlay.hwh]

puts "OVERLAY READY: [file join $out_dir gpt_overlay.bit]"
