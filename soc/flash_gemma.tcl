# flash_gemma.tcl - program the PL bitstream, load fp16 weights to DDR, run the app.
# Run: xsdb flash_gemma.tcl <ps7_init.tcl> <gemma_soc.bit> <model.bin> <gemma_app.elf>
set ps7init [lindex $argv 0]
set bit     [lindex $argv 1]
set model   [lindex $argv 2]
set elf     [lindex $argv 3]
puts "ps7_init : $ps7init"
puts "bitstream: $bit"
puts "weights  : $model  -> DDR 0x10000000"
puts "elf      : $elf"

connect

if {![catch {targets -set -nocase -filter {name =~ "*Cortex-A9*#1"}}]} { catch {stop} }

targets -set -nocase -filter {name =~ "*Cortex-A9*#0"}
catch {stop}
rst -processor
source $ps7init
ps7_init
ps7_post_config

targets -set -nocase -filter {name =~ "*xc7z010*" || name =~ "*PL*"}
fpga -file $bit

targets -set -nocase -filter {name =~ "*Cortex-A9*#0"}
dow -data $model 0x10000000

dow $elf
con

puts "Running - open the J24 UART (115200 8N1); the PL should print a story."
disconnect
