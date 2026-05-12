# run_hls.tcl — Build sumcheck IP (v9)
#
# Usage:
#   cd hls_src/
#   vitis_hls -f run_hls.tcl

open_project sumcheck_hls_proj -reset
set_top sumcheck_kernel
add_files sumcheck_round.cpp
add_files sumcheck_round.h
add_files -tb sumcheck_tb.cpp

open_solution "solution1" -reset
set_part {xck26-sfvc784-2LV-c}
create_clock -period "5ns" -name default

puts "── C Simulation ──"
csim_design

puts "── C Synthesis ──"
csynth_design

puts "── C/RTL Co-simulation ──"
cosim_design -tool xsim -rtl verilog

puts "── Export IP ──"
export_design -format ip_catalog -description "Sumcheck Kernel v9" -display_name "sumcheck_kernel"

puts "Done. IP at: sumcheck_hls_proj/solution1/impl/ip/"
exit
