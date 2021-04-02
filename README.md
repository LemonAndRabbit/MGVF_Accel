# MGVF_Accel

## Notes 

This repository is adapted from https://github.com/SFU-HiAccel/rodinia-hls.git : MGVF Algorithm.

Revised to utilized multiple HBM banks.

All ports is assigned to HBM[0], but can be easily changed in `settings.cfg` , waiting for optimization.

## Methodology

+ [compute_overlap](compute_overlap)
  + Use compute overlap to avoid data transfer between kernels
+ edge_data_transfer: [exchange_hbm](exchange_hbm)
  + Use additional HBM banks to transfer edge data
  + Use depth=1 AXIS port to synchronize
+ edge_data_transfer: [exchange_stream](exchange_stream)
  + Use AXIS port to exchange edge data
+ compute_overlap kernels under a top function: [gathered_compute_overlap](gathered_compute_overlap)
+ hbm_exchange kernels under a top function: [gathered_exchange_hbm](gathered_exchange_hbm)
  + Use top_function to synchronize

All Initiation Interval has been set to 1 

## Verification Environment

*Only sw_emu is tested!*

+ **Host OS**
  + Ubuntu 20.04.1

+ **TARGET FPGA** 
  + Xilinx Alveo U50: xilinx_u50_gen3x16_xdma_201920_3

+ **Software Tools**
  + Vitis 2020.2
  + Xilinx Runtime(XRT) 2020.2