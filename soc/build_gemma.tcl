# build_gemma.tcl - PS7 + gemma_top accelerator -> bitstream for Z-turn Lite.
#   vivado -mode batch -source build_gemma.tcl
# Topology: PS M_AXI_GP0 -> AXI-Lite control (token/pos/start/logits),
#           gemma M_AXI  -> PS S_AXI_HP0 (streams fp16 weights from DDR),
#           PS UART1 (MIO48/49 = J24) for stdout, FCLK_CLK0 = 100 MHz.

set proj_dir [file dirname [file normalize [info script]]]
set rtl      [file normalize [file join $proj_dir .. rtl]]
set part     xc7z010clg400-1
set bd       system
set build    [file join $proj_dir build]

file delete -force $build
file mkdir $build
create_project gemma_soc $build -part $part -force

# ---- sources: fp16 core + wrapper (SV impl + Verilog BD wrapper) ----
add_files -norecurse [list \
    [file join $rtl fp16_mul_p.sv]   [file join $rtl fp16_add_p.sv] \
    [file join $rtl int2fp16.sv]     [file join $rtl fp2int_round16.sv] \
    [file join $rtl fp16_exp_p.sv]   [file join $rtl fp16_rsqrt_p.sv] \
    [file join $rtl pmac16.sv]       [file join $rtl axi_rd_m16.sv] \
    [file join $rtl gemma_fwd_p.sv]  [file join $rtl gemma_top.sv] \
    [file join $proj_dir gemma_top_v.v]]
set_property file_type SystemVerilog [get_files [file join $rtl *.sv]]
update_compile_order -fileset sources_1

# ---- block design ----
create_bd_design $bd
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
source [file join [file normalize [file join $proj_dir .. .. pl_uart]] z-turn-lite.tcl]
set cfg [apply_preset 0]
dict for {k v} $cfg { catch {set_property $k $v $ps} }

set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100}] $ps

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} \
    [get_bd_cells ps7]

set acc [create_bd_cell -type module -reference gemma_top_v gemma_0]

set rstg [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rstgen]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0]    [get_bd_pins rstgen/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rstgen/ext_reset_in]

set sc_ctl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $sc_ctl
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0]  [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI] [get_bd_intf_pins gemma_0/s_axil]

set sc_hp [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 sc_hp]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $sc_hp
connect_bd_intf_net [get_bd_intf_pins gemma_0/m_axi]  [get_bd_intf_pins sc_hp/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hp/M00_AXI]  [get_bd_intf_pins ps7/S_AXI_HP0]

set fclk [get_bd_pins ps7/FCLK_CLK0]
foreach p {ps7/M_AXI_GP0_ACLK ps7/S_AXI_HP0_ACLK sc_ctl/aclk sc_hp/aclk gemma_0/clk} {
    connect_bd_net $fclk [get_bd_pins $p]
}
set arstn [get_bd_pins rstgen/peripheral_aresetn]
foreach p {sc_ctl/aresetn sc_hp/aresetn gemma_0/resetn} {
    connect_bd_net $arstn [get_bd_pins $p]
}

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

foreach seg [get_bd_addr_segs -of_objects [get_bd_intf_pins ps7/M_AXI_GP0]] {
    puts "CTRL_SEG $seg  offset=[get_property OFFSET $seg]  range=[get_property RANGE $seg]"
}

if {[lindex $argv 0] eq "validate"} { puts "VALIDATE_OK"; return }

set bd_file [get_files ${bd}.bd]
make_wrapper -files $bd_file -top
add_files -norecurse [file join $build gemma_soc.gen sources_1 bd $bd hdl ${bd}_wrapper.v]
set_property top ${bd}_wrapper [current_fileset]
generate_target all $bd_file
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 10
wait_on_run synth_1

set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 10
wait_on_run impl_1

# ---- utilization + timing reports (gate counts / Fmax) ----
open_run impl_1
report_utilization        -file [file join $proj_dir util.rpt]
report_timing_summary     -file [file join $proj_dir timing.rpt]
set wns [get_property STATS.WNS [get_runs impl_1]]
set tns [get_property STATS.TNS [get_runs impl_1]]
puts "TIMING: WNS=$wns ns  TNS=$tns ns  (clock target 100 MHz / 10 ns)"
puts "UTIL report -> [file join $proj_dir util.rpt]"

set bit [glob -nocomplain [file join $build gemma_soc.runs impl_1 *.bit]]
if {$bit eq ""} { error "bitstream not produced - check impl_1 logs" }
file copy -force $bit [file join $proj_dir gemma_soc.bit]
write_hw_platform -fixed -include_bit -force [file join $proj_dir gemma_soc.xsa]
puts "INFO: bitstream -> [file join $proj_dir gemma_soc.bit]"
puts "INFO: XSA       -> [file join $proj_dir gemma_soc.xsa]"
puts "BUILD_OK"
