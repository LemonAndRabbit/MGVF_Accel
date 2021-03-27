#ifndef LC_MGVF_H
#define LC_MGVF_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define GRID_ROWS 1024
#define GRID_COLS 128

#define PARTITION_FACTOR 2
#define PART_ROWS GRID_ROWS/PARTITION_FACTOR

#define TILE_ROWS 64

#define PARA_FACTOR 16


#define ITERATION 2

#define PART_ROWS GRID_ROWS/2 + ITERATION


#define TOP 0
#define BOTTOM (GRID_ROWS / TILE_ROWS - 1)


#define MU 0.5
#define LAMBDA (8.0 * MU + 1.0)
#define MU_O_LAMBDA  (MU / LAMBDA)
#define ONE_O_LAMBDA (1.0 / LAMBDA)


#define COVERAGE 0.000000000001
#define MAX_RADIUS 1

#define TYPE float

struct bench_args_t {
    float imgvf[GRID_ROWS * GRID_COLS];
    float I[GRID_ROWS * GRID_COLS];
};

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



#endif
