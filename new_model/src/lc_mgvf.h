#ifndef LC_MGVF_H
#define LC_MGVF_H

#define GRID_ROWS 1024
#define GRID_COLS 1024

#define PARA_FACTOR 16

#define KERNEL_COUNT 2
#define PART_ROWS GRID_ROWS / KERNEL_COUNT

#define ITERATION 2

#define TOP_ROWS GRID_ROWS/2 + ITERATION

#define MU 0.5
#define LAMBDA (8.0 * MU + 1.0)
#define MU_O_LAMBDA  (MU / LAMBDA)
#define ONE_O_LAMBDA (1.0 / LAMBDA)

#define COALESCING_5_512bit
#ifdef COALESCING_5_512bit
//#include <gmp.h>
//#define __gmp_const const
#include "ap_int.h"
#include <inttypes.h>
    const int DWIDTH = 512;
#define INTERFACE_WIDTH ap_uint<DWIDTH>
    const int WIDTH_FACTOR = DWIDTH/32;
#endif

#define INTERFACE_WIDTH_AUG ap_uint<DWIDTH + 64>

#endif