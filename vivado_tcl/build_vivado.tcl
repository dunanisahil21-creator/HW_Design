# build_vivado.tcl — Create Vivado project, block design, and bitstream for KV260
#
# Usage:
#   cd /path/to/sumcheck_fpga/vivado_tcl
#   vivado -mode batch -source build_vivado.tcl
#
# Prerequisites:
#   - Vitis HLS IP already exported to ../hls_src/sumcheck_hls_proj/solution1/impl/ip/
#   - Vivado 2022.1 (or compatible) with KV260 board files installed
#
# Output:
#   - sumcheck_kv260/sumcheck_kv260.runs/impl_1/system_wrapper.bit
#   - sumcheck_kv260/sumcheck_kv260.gen/sources_1/bd/system/hw_handoff/system.hwh
#
# What this script does (replaces the full GUI walkthrough):
#   1) Create project targeting KV260 board
#   2) Add HLS IP repo
#   3) Create block design with Zynq PS + sumcheck IP
#   4) Configure PS interfaces (AXI HP, AXI LPD, interrupts)
#   5) Run connection automation
#   6) Wire interrupt
#   7) Validate, wrap, synthesize, implement, generate bitstream
#   8) Copy .bit + .hwh to output directory

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — edit these paths for your setup
# ═══════════════════════════════════════════════════════════════════════════

# Where the HLS IP was exported
set HLS_IP_DIR  [file normalize "../hls_src/sumcheck_hls_proj/solution1/impl/ip"]

# Project output directory
set PROJ_DIR    "sumcheck_kv260"
set PROJ_NAME   "sumcheck_kv260"

# Number of parallel jobs for synthesis/implementation
set NUM_JOBS    8

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Create project
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 1: Creating Vivado project ────"

create_project ${PROJ_NAME} ./${PROJ_DIR} -part xck26-sfvc784-2LV-c -force

# Set board part (KV260 Vision AI Starter Kit)
# This configures board presets for DDR4, QSPI, etc.
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Add HLS IP to repository
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 2: Adding HLS IP repository ────"

if {![file exists $HLS_IP_DIR]} {
    puts "ERROR: HLS IP directory not found at: $HLS_IP_DIR"
    puts "       Run the Vitis HLS build first (run_hls.tcl)"
    exit 1
}

set_property ip_repo_paths [list $HLS_IP_DIR] [current_fileset]
update_ip_catalog -rebuild

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Create block design
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 3: Creating block design ────"

create_bd_design "system"

# ─── 3a: Add Zynq UltraScale+ MPSoC ───
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_ultra_ps_e_0

# Apply board presets (DDR4, QSPI, clocking for K26 SOM)
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} \
    [get_bd_cells zynq_ultra_ps_e_0]

# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Configure PS interfaces
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 4: Configuring PS interfaces ────"

# Configure the Zynq PS with the interfaces we need:
#   - M_AXI_HPM0_LPD (32-bit): AXI-Lite master for controlling the IP's registers
#   - S_AXI_HP0_FPD  (128-bit): High-performance slave for IP's DDR access
#   - PL interrupt IRQ0
#
# We disable FPD masters since we don't need them in this design.
# The LPD master runs at a lower power domain — good for control traffic.
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0        {0}   \
    CONFIG.PSU__USE__M_AXI_GP1        {1}   \
    CONFIG.PSU__MAXIGP1__DATA_WIDTH   {32}  \
    CONFIG.PSU__USE__M_AXI_GP2        {0}   \
    CONFIG.PSU__USE__S_AXI_GP2        {1}   \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH   {128} \
    CONFIG.PSU__USE__IRQ0             {1}   \
] [get_bd_cells zynq_ultra_ps_e_0]

# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Add sumcheck HLS IP
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 5: Adding sumcheck IP ────"

# Find the IP — the VLNV may vary based on your HLS export settings.
# Common pattern: xilinx.com:hls:sumcheck_prove_round:1.0
# If this fails, run `get_ipdefs *sumcheck*` in Vivado Tcl to find the exact VLNV.
set ip_vlnv [lindex [get_ipdefs *sumcheck_prove_round*] 0]
if {$ip_vlnv eq ""} {
    puts "ERROR: sumcheck_prove_round IP not found in repository."
    puts "       Check that the HLS export completed and the path is correct."
    puts "       Try running: get_ipdefs *sumcheck*"
    exit 1
}
puts "  Found IP: $ip_vlnv"
create_bd_cell -type ip -vlnv $ip_vlnv sumcheck_0

# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Connect AXI-Lite (control interface)
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 6: Connecting AXI-Lite control ────"

# Connect PS LPD master → sumcheck IP's s_axi_control
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { \
        Clk_master {Auto} \
        Clk_slave  {Auto} \
        Clk_xbar   {Auto} \
        Master     {/zynq_ultra_ps_e_0/M_AXI_HPM0_LPD} \
        Slave      {/sumcheck_0/s_axi_control} \
        intc_ip    {New AXI Interconnect} \
        master_apm {0} \
    } [get_bd_intf_pins sumcheck_0/s_axi_control]

# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Connect AXI-Master (DDR access for table reads/writes)
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 7: Connecting AXI-Master to DDR ────"

# Connect sumcheck IP's m_axi_gmem0 → PS HP0 slave (DDR)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { \
        Clk_master {Auto} \
        Clk_slave  {Auto} \
        Clk_xbar   {Auto} \
        Master     {/sumcheck_0/m_axi_gmem0} \
        Slave      {/zynq_ultra_ps_e_0/S_AXI_HP0_FPD} \
        intc_ip    {New AXI SmartConnect} \
        master_apm {0} \
    } [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]

# ═══════════════════════════════════════════════════════════════════════════
# Step 8: Connect interrupt
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 8: Connecting interrupt ────"

# Wire the IP's interrupt output to the PS pl_ps_irq0 input
connect_bd_net [get_bd_pins sumcheck_0/interrupt] \
               [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

# ═══════════════════════════════════════════════════════════════════════════
# Step 9: Assign addresses
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 9: Assigning addresses ────"

assign_bd_address
# The auto-assign gives:
#   s_axi_control → some offset in 0xA000_0000 range (read from PYNQ)
#   m_axi_gmem0   → full DDR range 0x0 to 0x7FFFFFFF

# ═══════════════════════════════════════════════════════════════════════════
# Step 10: Validate design
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 10: Validating design ────"

validate_bd_design
save_bd_design

# ═══════════════════════════════════════════════════════════════════════════
# Step 11: Create HDL wrapper
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 11: Creating HDL wrapper ────"

make_wrapper -files [get_files system.bd] -top
set wrapper_file [glob ${PROJ_DIR}/${PROJ_NAME}.gen/sources_1/bd/system/hdl/system_wrapper.v]
add_files -norecurse $wrapper_file
update_compile_order -fileset sources_1

# ═══════════════════════════════════════════════════════════════════════════
# Step 12: Synthesis
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 12: Running Synthesis ────"

launch_runs synth_1 -jobs $NUM_JOBS
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════════
# Step 13: Implementation + Bitstream
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 13: Running Implementation + Bitstream ────"

launch_runs impl_1 -to_step write_bitstream -jobs $NUM_JOBS
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════════
# Step 14: Copy output files for PYNQ
# ═══════════════════════════════════════════════════════════════════════════
puts "──── Step 14: Collecting overlay files ────"

file mkdir ../pynq_overlay

# .bit file
set bit_file [glob ${PROJ_DIR}/${PROJ_NAME}.runs/impl_1/system_wrapper.bit]
file copy -force $bit_file ../pynq_overlay/sumcheck.bit
puts "  Copied: $bit_file → ../pynq_overlay/sumcheck.bit"

# .hwh file
set hwh_file [glob ${PROJ_DIR}/${PROJ_NAME}.gen/sources_1/bd/system/hw_handoff/system.hwh]
file copy -force $hwh_file ../pynq_overlay/sumcheck.hwh
puts "  Copied: $hwh_file → ../pynq_overlay/sumcheck.hwh"

# ═══════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════
puts ""
puts "══════════════════════════════════════════════════════════════"
puts " BUILD COMPLETE"
puts ""
puts " Overlay files ready for PYNQ:"
puts "   ../pynq_overlay/sumcheck.bit"
puts "   ../pynq_overlay/sumcheck.hwh"
puts ""
puts " Copy both files to the KV260:"
puts "   scp ../pynq_overlay/sumcheck.* ubuntu@<kv260-ip>:~/jupyter_notebooks/"
puts ""
puts " Timing summary:"
set wns [get_property STATS.WNS [get_runs impl_1]]
set tns [get_property STATS.TNS [get_runs impl_1]]
puts "   WNS (worst negative slack): ${wns} ns"
puts "   TNS (total negative slack):  ${tns} ns"
if {$wns < 0} {
    puts "   WARNING: Timing violated! Consider reducing clock frequency."
}
puts "══════════════════════════════════════════════════════════════"

exit
