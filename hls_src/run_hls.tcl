# run_hls.tcl — Build sumcheck IP from the command line
#
# Usage:
#   cd /path/to/sumcheck_fpga/hls_src
#   vitis_hls -f run_hls.tcl
#
# This script does everything the GUI does:
#   1) Creates a project targeting the KV260 part
#   2) Adds source + testbench files
#   3) Runs C simulation (csim)
#   4) Runs C synthesis
#   5) Runs C/RTL co-simulation
#   6) Exports the IP to the Vivado IP catalog format
#
# After running, the exported IP is at:
#   sumcheck_hls_proj/solution1/impl/ip/

# ── Project setup ──
open_project sumcheck_hls_proj -reset
set_top sumcheck_prove_round
add_files sumcheck_round.cpp
add_files sumcheck_round.h
add_files -tb sumcheck_tb.cpp

# ── Solution: target KV260 at 200 MHz ──
open_solution "solution1" -reset
set_part {xck26-sfvc784-2LV-c}
create_clock -period "5ns" -name default

# ── Step 1: C Simulation ──
puts "──────────────────────────────────"
puts " Running C Simulation (csim)..."
puts "──────────────────────────────────"
csim_design

# ── Step 2: C Synthesis ──
puts "──────────────────────────────────"
puts " Running C Synthesis..."
puts "──────────────────────────────────"
csynth_design

# ── Step 3: C/RTL Co-simulation (optional, comment out to save time) ──
# puts "──────────────────────────────────"
# puts " Running Co-simulation..."
# puts "──────────────────────────────────"
# cosim_design

# ── Step 4: Export RTL as IP ──
puts "──────────────────────────────────"
puts " Exporting IP..."
puts "──────────────────────────────────"
export_design -format ip_catalog -description "Sumcheck Prover Round" -display_name "sumcheck_prove_round"

puts ""
puts "══════════════════════════════════"
puts " Done! IP exported to:"
puts " sumcheck_hls_proj/solution1/impl/ip/"
puts "══════════════════════════════════"

exit
