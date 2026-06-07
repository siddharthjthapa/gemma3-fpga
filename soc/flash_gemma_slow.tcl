# flash_gemma_slow.tcl - same as flash_gemma.tcl but runs the PL at HALF clock
# (FCLK_CLK0 100 -> 50 MHz) by reprogramming the PS FPGA0 clock divider, to test
# whether the on-board glitches are SETUP-timing related (same bitstream, slower).
set ps7init [lindex $argv 0]
set bit     [lindex $argv 1]
set model   [lindex $argv 2]
set elf     [lindex $argv 3]

connect
if {![catch {targets -set -nocase -filter {name =~ "*Cortex-A9*#1"}}]} { catch {stop} }
targets -set -nocase -filter {name =~ "*Cortex-A9*#0"}
stop
rst -processor
source $ps7init
ps7_init
ps7_post_config

# ---- halve FCLK_CLK0: unlock SLCR, DIVISOR0 2->4 (0x200200 -> 0x200400) ----
mwr 0xF8000008 0xDF0D
mwr 0xF8000170 0x200400
puts "FCLK0_CTRL now = [mrd -value 0xF8000170]  (expect 0x200400 = 50 MHz)"

targets -set -nocase -filter {name =~ "*xc7z010*" || name =~ "*PL*"}
fpga -file $bit

targets -set -nocase -filter {name =~ "*Cortex-A9*#0"}
dow -data $model 0x10000000
dow $elf
con
puts "Running at 50 MHz - open J24 UART (115200 8N1)."
disconnect
