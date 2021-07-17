#include "lc_mgvf.h"
#include <fstream>
#define __kernel
#define __global

static float heaviside(float x) {
    // A simpler, faster approximation of the Heaviside function
    float out = 0.0;
    if ((float)x > (float)-0.0001) out = (float)0.5;
    if ((float)x >  (float)0.0001) out = (float)1.0;
    return out;
}

static float lc_mgvf_stencil_core(float c, float ul, float u, float ur, float l, float r, float dl, float d, float dr, float vI)
{
    float UL = ul - c;
    float U  = u  - c;
    float UR = ur - c;
    float L  = l  - c;
    float R  = r  - c;
    float DL = dl - c;
    float D  = d  - c;
    float DR = dr - c;

    float vHe = c + MU_O_LAMBDA * (heaviside(UL) * UL + heaviside(U) * U + heaviside(UR) * UR + heaviside(L) * L + heaviside(R) * R + heaviside(DL) * DL + heaviside(D) * D + heaviside(DR) * DR);

    float new_val = vHe - (ONE_O_LAMBDA * vI * (vHe - vI));

    return new_val;
}

static void lc_mgvf(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I, bool up_mode, bool down_mode)
{
    INTERFACE_WIDTH imgvf_rf[GRID_COLS/WIDTH_FACTOR * 2 + 2];
#pragma HLS array_partition variable=imgvf_rf complete dim=0

    INTERFACE_WIDTH_AUG up_buffer;
    INTERFACE_WIDTH_AUG middle_buffer;
    INTERFACE_WIDTH_AUG down_buffer;
    uint32_t imgvf_rf_poped_out;

    int i;

    const int input_bound = GRID_COLS/WIDTH_FACTOR * 2 + 2;
INITIALZE_LOOP:    
    for (i = 0; i < input_bound; i += 1) {
#pragma HLS pipeline II=1
        imgvf_rf[i] = imgvf[i];
    }

MAJOR_LOOP:
    for (i = 0; i < GRID_COLS / WIDTH_FACTOR * PART_ROWS; i++) {
#pragma HLS pipeline II=1        
        int k;

        up_buffer.range(31, 0) = imgvf_rf_poped_out;
        up_buffer.range(543, 32) = imgvf_rf[0];
        up_buffer.range(575, 544) = imgvf_rf[1].range(31, 0);

        middle_buffer.range(31, 0) = imgvf_rf[GRID_COLS/WIDTH_FACTOR - 1].range(511, 480);
        middle_buffer.range(543, 32) = imgvf_rf[GRID_COLS/WIDTH_FACTOR];
        middle_buffer.range(575, 544) = imgvf_rf[GRID_COLS/WIDTH_FACTOR + 1].range(31, 0);

        down_buffer.range(31, 0) = imgvf_rf[GRID_COLS/WIDTH_FACTOR*2 - 1].range(511, 480);
        down_buffer.range(543, 32) = imgvf_rf[GRID_COLS/WIDTH_FACTOR*2];
        down_buffer.range(575, 544) = imgvf_rf[GRID_COLS/WIDTH_FACTOR*2 + 1].range(31,0);        

COMPUTE_LOOP:
        for (k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll

            float ul[PARA_FACTOR], u[PARA_FACTOR], ur[PARA_FACTOR], l[PARA_FACTOR], c[PARA_FACTOR], r[PARA_FACTOR], dl[PARA_FACTOR], d[PARA_FACTOR], dr[PARA_FACTOR], vI[PARA_FACTOR];
#pragma HLS array_partition variable=ul complete dim=0
#pragma HLS array_partition variable=u complete dim=0
#pragma HLS array_partition variable=ur complete dim=0
#pragma HLS array_partition variable=l complete dim=0
#pragma HLS array_partition variable=c complete dim=0
#pragma HLS array_partition variable=r complete dim=0
#pragma HLS array_partition variable=dl complete dim=0
#pragma HLS array_partition variable=d complete dim=0
#pragma HLS array_partition variable=dr complete dim=0

            unsigned int idx_k = k<<5;
            unsigned int offset = idx_k + 32;

            bool is_top = up_mode & (i < GRID_COLS / WIDTH_FACTOR);
            bool is_right = (i % (GRID_COLS / WIDTH_FACTOR) == (GRID_COLS / WIDTH_FACTOR - 1)) && (k == WIDTH_FACTOR - 1);
            bool is_bottom = down_mode & (i >= GRID_COLS / WIDTH_FACTOR * (PART_ROWS  - 1));
            bool is_left = (i % (GRID_COLS / WIDTH_FACTOR) == 0) && (k == 0);

            //c[k] = imgvf_rf[GRID_COLS * (0) + 0 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_c = middle_buffer.range(offset + 31, offset);
            c[k] = *((float*)(&temp_c));

            //ul[k] = (is_top || is_left) ? c[k] : imgvf_rf[GRID_COLS * (-1) + -1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_ul = up_buffer.range(offset - 1, offset - 32);
            ul[k] = (is_top || is_left) ? c[k] : *((float*)(&temp_ul));

            //u[k] = (is_top) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 0 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_u = up_buffer.range(offset + 31, offset);
            u[k] = (is_top)? c[k] : *((float*)(&temp_u));

            //ur[k] = (is_top || is_right) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_ur = up_buffer.range(offset + 63, offset + 32);
            ur[k] = (is_top || is_right)? c[k] : *((float*)(&temp_ur));

            //l[k] = (is_left) ? c[k] : imgvf_rf[GRID_COLS * (0) + -1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_l = middle_buffer.range(offset - 1, offset - 32);
            l[k] = (is_left)? c[k] : *((float*)(&temp_l));
    
            //r[k] = (is_right) ? c[k] : imgvf_rf[GRID_COLS * (0) + 1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_r = middle_buffer.range(offset + 63, offset + 32);
            r[k] = (is_right)? c[k] : *((float*)(&temp_r));

            //dl[k] = (is_bottom || is_left) ? c[k] : imgvf_rf[GRID_COLS * (1) + -1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_dl = down_buffer.range(offset - 1, offset - 32);
            dl[k] = (is_bottom || is_left) ? c[k] : *((float*)(&temp_dl));

            //d[k] = (is_bottom) ? c[k] : imgvf_rf[GRID_COLS * (1) + 0 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_d = down_buffer.range(offset + 31, offset);
            d[k] = (is_bottom) ? c[k] : *((float*)(&temp_d));

            
            //dr[k] = (is_bottom || is_right) ? c[k] : imgvf_rf[GRID_COLS * (1) + 1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_dr = down_buffer.range(offset + 63, offset + 32);
            dr[k] = (is_bottom || is_right) ? c[k] : *((float*)(&temp_dr));

            uint32_t temp_I = I[i].range(idx_k+31, idx_k);
            float read_I = *((float*)(&temp_I));
            vI[k] = read_I;

            float res = lc_mgvf_stencil_core(c[k], ul[k], u[k], ur[k], l[k], r[k], dl[k], d[k], dr[k], vI[k]);

            result[i + GRID_COLS/WIDTH_FACTOR].range(idx_k+31, idx_k) = *((uint32_t *)(&res));
        }
/*
FILL_LOOP:
        for(;k < WIDTH_FACTOR; k++){
#pragma hls unroll            
            unsigned int idx = (i*WIDTH_FACTOR+k) / WIDTH_FACTOR;
            unsigned int range_idx =  (i*WIDTH_FACTOR+k) % WIDTH_FACTOR * 32;

            float res = 0;
            result[idx].range(range_idx+31, range_idx) = *((uint32_t *)(&res));            
        }
*/
        imgvf_rf_poped_out = imgvf_rf[0].range(511, 480);


DOWN_SHIFT_LOOP:
        for (k = 0; k < GRID_COLS/WIDTH_FACTOR*2 + 1; k++) {
#pragma HLS unroll
            imgvf_rf[k] = imgvf_rf[k + 1];
        }

        unsigned int idx = GRID_COLS/WIDTH_FACTOR * 2 + (i + 2);
        imgvf_rf[GRID_COLS/WIDTH_FACTOR*2 + 1] = imgvf[idx];
        
	}

    return;
}

extern "C" {
__kernel void workload(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I, unsigned int mode, unsigned iter)
{
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=result1
#pragma HLS INTERFACE m_axi port=imgvf offset=slave bundle=imgvf1
#pragma HLS INTERFACE m_axi port=I offset=slave bundle=I1

#pragma HLS INTERFACE s_axilite port=result
#pragma HLS INTERFACE s_axilite port=imgvf
#pragma HLS INTERFACE s_axilite port=I

#pragma HLS INTERFACE s_axilite port=return

    int i;
    for(i=0; i<iter/2; i++){
        lc_mgvf(result, imgvf, I, mode==1 || mode==3, mode==2 || mode==3);
        lc_mgvf(imgvf, result, I, mode==1 || mode==3, mode==2 || mode==3);
    }
    return;
}
}
