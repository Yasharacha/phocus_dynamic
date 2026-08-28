# Makefile — Vitis build flow for tv_cfl_kernel
#
# Targets
#   csim      : compile kernel + testbench, run C-simulation (no HLS licence needed)
#   synth     : Vitis HLS synthesis only  (generates timing / resource report)
#   sw_emu    : software emulation   (fast, XRT required)
#   hw_emu    : hardware emulation   (RTL simulation via Vivado, slow)
#   hw        : full bitstream build (30–60 min, FPGA hardware required)
#   run_sw    : run the host binary against sw_emu xclbin
#   clean     : remove all build artefacts
#
# Prerequisites
#   source /tools/Xilinx/Vitis/2023.2/settings64.sh
#   source /opt/xilinx/xrt/setup.sh
#
# Adjust PLATFORM to match your Alveo card:
#   xilinx_u200_gen3x16_xdma_2_202110_1
#   xilinx_u250_gen3x16_xdma_4_1_202210_1
#   xilinx_u280_gen3x16_xdma_1_202211_1
#   xilinx_u50_gen3x16_xdma_5_202210_1

PLATFORM  ?= xilinx_u200_gen3x16_xdma_2_202110_1
KERNEL_SRC = ../hls/tv_cfl_kernel.cpp
KERNEL_INC = ../hls
KERNEL_NAME = tv_cfl_kernel
HOST_SRC   = ../host/host_tv_cfl.cpp
CONN_CFG   = connectivity.cfg
XCLBIN     = tv_cfl.xclbin
HOST_BIN   = host_tv_cfl

XRT_INC    ?= $(XILINX_XRT)/include
XRT_LIB    ?= $(XILINX_XRT)/lib

# ---- C-simulation (no Vitis licence) ----------------------------------------
# Uses g++ directly — quick correctness check on a workstation.

csim:
	g++ -O2 -std=c++17 -DHLS_CSIM \
	    -I$(KERNEL_INC) \
	    -o tb_tv_cfl \
	    ../hls/tb_tv_cfl.cpp \
	    $(KERNEL_SRC)
	./tb_tv_cfl

# ---- Vitis HLS synthesis only -----------------------------------------------
# Produces a timing / resource report in _hls_build/tv_cfl_kernel/syn/report/

synth:
	vitis_hls run_synth.tcl

run_synth.tcl:
	@echo "open_project _hls_build"                              >  $@
	@echo "set_top tv_cfl_kernel"                                >> $@
	@echo "add_files $(KERNEL_SRC) -cflags \"-I$(KERNEL_INC)\"" >> $@
	@echo "add_files -tb ../hls/tb_tv_cfl.cpp -cflags \"-I$(KERNEL_INC)\"" >> $@
	@echo "open_solution sol1 -flow_target vitis"                >> $@
	@echo "set_part {xcu200-fsgd2104-2-e}"                       >> $@
	@echo "create_clock -period 3.3 -name default"               >> $@
	@echo "csynth_design"                                        >> $@
	@echo "exit"                                                 >> $@

# ---- Software emulation build -----------------------------------------------

sw_emu: $(XCLBIN)_sw_emu

$(XCLBIN)_sw_emu:
	v++ -c -t sw_emu \
	    --platform $(PLATFORM) \
	    --kernel $(KERNEL_NAME) \
	    -I$(KERNEL_INC) \
	    -o tv_cfl_kernel_sw_emu.xo \
	    $(KERNEL_SRC)
	v++ -l -t sw_emu \
	    --platform $(PLATFORM) \
	    --config $(CONN_CFG) \
	    -o tv_cfl_sw_emu.xclbin \
	    tv_cfl_kernel_sw_emu.xo

# ---- Hardware emulation build -----------------------------------------------

hw_emu: $(XCLBIN)_hw_emu

$(XCLBIN)_hw_emu:
	v++ -c -t hw_emu \
	    --platform $(PLATFORM) \
	    --kernel $(KERNEL_NAME) \
	    -I$(KERNEL_INC) \
	    -o tv_cfl_kernel_hw_emu.xo \
	    $(KERNEL_SRC)
	v++ -l -t hw_emu \
	    --platform $(PLATFORM) \
	    --config $(CONN_CFG) \
	    -o tv_cfl_hw_emu.xclbin \
	    tv_cfl_kernel_hw_emu.xo

# ---- Full hardware build (bitstream) ----------------------------------------

hw:
	v++ -c -t hw \
	    --platform $(PLATFORM) \
	    --kernel $(KERNEL_NAME) \
	    -I$(KERNEL_INC) \
	    --advanced.prop kernel.$(KERNEL_NAME).kernel_flags="-std=c++14" \
	    -o tv_cfl_kernel_hw.xo \
	    $(KERNEL_SRC)
	v++ -l -t hw \
	    --platform $(PLATFORM) \
	    --config $(CONN_CFG) \
	    --advanced.param compiler.worstNegativeSlack=-0.1 \
	    -o tv_cfl_hw.xclbin \
	    tv_cfl_kernel_hw.xo

# ---- Host binary ------------------------------------------------------------

host:
	g++ -O2 -std=c++17 \
	    -I$(XRT_INC) \
	    -o $(HOST_BIN) \
	    $(HOST_SRC) \
	    -L$(XRT_LIB) -lxrt_coreutil -pthread

# ---- Run in software emulation ----------------------------------------------

run_sw: sw_emu host
	XCL_EMULATION_MODE=sw_emu ./$(HOST_BIN) tv_cfl_sw_emu.xclbin 0 100000 25

# ---- Run on real hardware ---------------------------------------------------

run_hw: hw host
	./$(HOST_BIN) tv_cfl_hw.xclbin 0 100000 25

# ---- Clean ------------------------------------------------------------------

clean:
	rm -rf _hls_build _x .run *.xo *.xclbin *.ltx *.log *.jou \
	       tb_tv_cfl $(HOST_BIN) run_synth.tcl xsim.dir
