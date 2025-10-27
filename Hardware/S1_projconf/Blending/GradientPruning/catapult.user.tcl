set TOT_PATH $env(TOT_PATH_CAT)
puts "sourcing global setting file ${TOT_PATH}S0_scripts/hls/catapult.global.tcl"
source ${TOT_PATH}S0_scripts/hls/catapult.global.tcl

proc hls::user_global_directives {} {}
proc hls::user_pre_compile {} { directive set -CLOCK_OVERHEAD 0.000000 }
proc hls::user_pre_assem {} {}
proc hls::setup_clocks {CLK_PERIOD} {
    directive set -CLOCK_NAME clk
    directive set -CLOCKS "clk \"-CLOCK_PERIOD $CLK_PERIOD\""
}
proc hls::user_pre_arch {} {}
proc hls::user_pre_extract {} {}

hls::do_catapult


