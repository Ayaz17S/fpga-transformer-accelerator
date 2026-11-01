# Build a Vivado GUI project with every RTL file, testbench and .mem file
# already added and in the right roles. Usage (from the project root):
#   vivado -mode gui -source syn/create_project.tcl
#   vivado -mode gui -source syn/create_project.tcl -tclargs <other part>
# The project lands in vivado/ (git-ignored) and is rebuilt from scratch each
# run, so the RTL files in rtl/ and tb/ stay the only source of truth.
# Default part is the PYNQ-Z2's Zynq-7020.

set part xc7z020clg400-1
if {$argc > 0} { set part [lindex $argv 0] }

set root [file normalize [file join [file dirname [info script]] ..]]

create_project fpga_accelerator [file join $root vivado] -part $part -force

# Design sources. The .mem files go here too: synthesis needs them for
# $readmemh, and Vivado copies them into the simulation run directory.
add_files -fileset sources_1 [glob [file join $root rtl *.sv] [file join $root rtl *.v]]
add_files -fileset sources_1 [glob [file join $root data *.mem]]

# Simulation-only sources: testbenches and tb_pkg.
add_files -fileset sim_1 [glob [file join $root tb *.sv]]

# Tops: change the simulation top in Sources -> Simulation Sources ->
# right-click a testbench -> Set as Top.
set_property top gpt_core    [get_filesets sources_1]
set_property top tb_gpt_core [get_filesets sim_1]
set_property -name {xsim.simulate.runtime} -value {all} -objects [get_filesets sim_1]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

puts "Project ready: [file join $root vivado fpga_accelerator.xpr]"
