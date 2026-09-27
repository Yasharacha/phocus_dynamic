# build_bd.tcl - Vivado 2025.1 block design for tv_cfl_kernel on the Ultra96 (ZU3EG).
#
# Usage (from a short path, e.g. C:/v, because the HLS IP has long file names):
#   vivado -mode batch -source build_bd.tcl                 ;# create + validate only
#   vivado -mode batch -source build_bd.tcl -tclargs bitstream   ;# ...and build the bitstream
#
# Prerequisite: the HLS IP package extracted to $ip_repo
# (unzip hls/hls_out/hls/impl/ip/xilinx_com_hls_tv_cfl_kernel_1_0.zip there).
#
# Design: zynq_ps.M_AXI_HPM0_FPD -> ic_ctl -> kernel.s_axi_control
#         kernel.m_axi_gmem0/gmem1 -> ic_mem -> zynq_ps.S_AXI_HP0_FPD
# PL clock: pl_clk0 at $pl_mhz MHz (set below). The PS DDR/MIO setup is done by the FSBL on the board's
# SD card at boot, so only the PL-facing settings matter here.

# Build on D: (C: was nearly full, which crashed Vivado's helper processes).
set proj_dir D:/v/tvcfl
set ip_repo  D:/v/ip
set out_dir  D:/v/out
# PL clock in MHz. 150 MHz = IOPLL/10 (the PS also offers 100 = /15 and 187.5 = /8).
set pl_mhz 150
set out_name tv_cfl_buckley150   ;# output file base name in $out_dir
set do_bitstream [expr {[lsearch $argv bitstream] >= 0}]

# "coherent" on the command line: attach the kernel to the PS cache-coherent port HPC0 instead of the
# non-coherent HP0, and mark its AXI requests cacheable + shareable, so the FPGA reads the CPU's
# cache-resident data and the CPU no longer has to flush the array before every check.
set coherent [expr {[lsearch $argv coherent] >= 0}]
if {$coherent} {
    set ps_use_gp    GP0                  ;# S_AXI_HPC0_FPD
    set ps_intf      S_AXI_HPC0_FPD
    set ps_clk_pin   saxihpc0_fpd_aclk
    set out_name     ${out_name}_coh
} else {
    set ps_use_gp    GP2                  ;# S_AXI_HP0_FPD
    set ps_intf      S_AXI_HP0_FPD
    set ps_clk_pin   saxihp0_fpd_aclk
}

create_project tvcfl $proj_dir -part xczu3eg-sbva484-1-e -force
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

create_bd_design design_1

# ---- Processing system ----
set ps_vlnv [lindex [lsort -decreasing [get_ipdefs xilinx.com:ip:zynq_ultra_ps_e:*]] 0]
set ps [create_bd_cell -type ip -vlnv $ps_vlnv zynq_ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_$ps_use_gp {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $pl_mhz \
] $ps

# ---- Kernel ----
set k_vlnv [lindex [lsort -decreasing [get_ipdefs xilinx.com:hls:tv_cfl_kernel:*]] 0]
set k [create_bd_cell -type ip -vlnv $k_vlnv tv_cfl_kernel_0]
if {$coherent} {
    # AxCACHE = 1111 (write-back, read/write-allocate) and AxUSER = 1 (shareable) on both memory ports.
    foreach {g uw} {GMEM0 16 GMEM1 8} {   ;# uw = R/W user width: one bit per data byte (128-bit / 64-bit port)
        set_property -dict [list \
            CONFIG.C_M_AXI_${g}_CACHE_VALUE {"1111"} \
            CONFIG.C_M_AXI_${g}_USER_VALUE {0x00000001} \
            CONFIG.C_M_AXI_${g}_ENABLE_USER_PORTS {true} CONFIG.C_M_AXI_${g}_RUSER_WIDTH $uw CONFIG.C_M_AXI_${g}_WUSER_WIDTH $uw \
        ] $k
    }
}

# ---- Interconnects and reset ----
# SmartConnect: the classic axi_interconnect is unsupported in Vivado 2025.1.
set sc_vlnv [lindex [lsort -decreasing [get_ipdefs xilinx.com:ip:smartconnect:*]] 0]
set ic_ctl [create_bd_cell -type ip -vlnv $sc_vlnv ic_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $ic_ctl
set ic_mem [create_bd_cell -type ip -vlnv $sc_vlnv ic_mem]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $ic_mem
set rst [create_bd_cell -type ip -vlnv [lindex [lsort -decreasing [get_ipdefs xilinx.com:ip:proc_sys_reset:*]] 0] rst_ps]

# ---- Clocks and resets ----
set clk [get_bd_pins zynq_ps/pl_clk0]
connect_bd_net $clk \
    [get_bd_pins zynq_ps/maxihpm0_fpd_aclk] \
    [get_bd_pins zynq_ps/$ps_clk_pin] \
    [get_bd_pins tv_cfl_kernel_0/ap_clk] \
    [get_bd_pins rst_ps/slowest_sync_clk] \
    [get_bd_pins ic_ctl/aclk] [get_bd_pins ic_mem/aclk]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins rst_ps/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps/peripheral_aresetn] \
    [get_bd_pins tv_cfl_kernel_0/ap_rst_n] \
    [get_bd_pins ic_ctl/aresetn] [get_bd_pins ic_mem/aresetn]

# ---- AXI connections ----
connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD]      [get_bd_intf_pins ic_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ic_ctl/M00_AXI]              [get_bd_intf_pins tv_cfl_kernel_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins tv_cfl_kernel_0/m_axi_gmem0] [get_bd_intf_pins ic_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins tv_cfl_kernel_0/m_axi_gmem1] [get_bd_intf_pins ic_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins ic_mem/M00_AXI]              [get_bd_intf_pins zynq_ps/$ps_intf]

# ---- Addresses ----
assign_bd_address
puts "==== ADDRESS MAP ===="
foreach seg [get_bd_addr_segs] {
    puts "[get_property NAME $seg]  offset=[get_property OFFSET $seg]  range=[get_property RANGE $seg]"
}

validate_bd_design
save_bd_design

# ---- Wrapper ----
make_wrapper -files [get_files $proj_dir/tvcfl.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse $proj_dir/tvcfl.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

if {$do_bitstream} {
    # One job at a time: parallel jobs crashed Vivado (EXCEPTION_ACCESS_VIOLATION) on a
    # 16 GB machine, suspected memory pressure.
    launch_runs impl_1 -to_step write_bitstream -jobs 1
    wait_on_run impl_1
    file mkdir $out_dir
    file copy -force $proj_dir/tvcfl.runs/impl_1/design_1_wrapper.bit $out_dir/${out_name}.bit
    write_hwdef -force -file $out_dir/${out_name}.hdf
    catch {file copy -force [glob $proj_dir/tvcfl.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh] $out_dir/${out_name}.hwh}
    puts "BITSTREAM DONE: $out_dir"
}
puts "SCRIPT FINISHED OK"
