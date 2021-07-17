#include "lc_mgvf.h"

#define __kernel
#define __global

#define PART_ROWS GRID_ROWS/2

static float heaviside(float x) {
    // A simpler, faster approximation of the Heaviside function
    float out = 0.0;
    if ((float)x > (float)-0.0001) out = (float)0.5;
    if ((float)x >  (float)0.0001) out = (float)1.0;
    return out;
}

static float lc_mgvf_stencil_core(float c, float ul, float u, float ur, float l, float r, float dl, float d, float dr, float vI)
{
    float UL = (float)ul - (float)c;
    float U  = (float)u  - (float)c;
    float UR = (float)ur - (float)c;
    float L  = (float)l  - (float)c;
    float R  = (float)r  - (float)c;
    float DL = (float)dl - (float)c;
    float D  = (float)d  - (float)c;
    float DR = (float)dr - (float)c;

    float vHe = (float)c + (float)MU_O_LAMBDA * (float)(
            (float)heaviside(UL) * (float)UL +
            (float)heaviside(U)  * (float)U  +
            (float)heaviside(UR) * (float)UR +
            (float)heaviside(L)  * (float)L  +
            (float)heaviside(R)  * (float)R  +
            (float)heaviside(DL) * (float)DL +
            (float)heaviside(D)  * (float)D  +
            (float)heaviside(DR) * (float)DR
    );

    float new_val = (float)vHe - ((float)ONE_O_LAMBDA * (float)vI * (float)((float)vHe - (float)vI));

    return new_val;
}

static void lc_mgvf(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I, unsigned int bonus)
{
    INTERFACE_WIDTH imgvf_rf[GRID_COLS/WIDTH_FACTOR * 2 + 2];
#pragma HLS array_partition variable=imgvf_rf complete dim=0
    int i;

    int input_bound = GRID_COLS/WIDTH_FACTOR * 2 + 2;
DOWN_INITIALZE_LOOP:    
    for (i = 0; i < input_bound; i += 1) {
#pragma HLS pipeline II=1
        imgvf_rf[i] = imgvf[i];
    }

    uint32_t imgvf_rf_poped1;
    uint32_t imgvf_rf_poped2;
    uint32_t imgvf_rf_poped3;
    uint32_t imgvf_rf_poped4;
    uint32_t imgvf_rf_poped5;
    uint32_t imgvf_rf_poped6;

DOWN_MAJOR_LOOP:
    for (i = 0; i < GRID_COLS / WIDTH_FACTOR * (PART_ROWS+ITERATION); i++) {
#pragma HLS pipeline II=1        
        int k;
    
    imgvf_rf_poped2 = imgvf_rf[1].range(31, 0);
    imgvf_rf_poped3 = imgvf_rf[GRID_COLS/WIDTH_FACTOR - 1].range(511, 480);
    imgvf_rf_poped4 = imgvf_rf[GRID_COLS/WIDTH_FACTOR + 1].range(31, 0);
    imgvf_rf_poped5 = imgvf_rf[GRID_COLS/WIDTH_FACTOR*2 - 1].range(511, 480);
    imgvf_rf_poped6 = imgvf_rf[GRID_COLS/WIDTH_FACTOR*2 + 1].range(31,0);    

DOWN_COMPUTE_LOOP:
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

            int is_top = false;
            int is_right = (i % (GRID_COLS / WIDTH_FACTOR) == (GRID_COLS / WIDTH_FACTOR - 1)) && (k == WIDTH_FACTOR - 1);
            int is_bottom = (i >= GRID_COLS / WIDTH_FACTOR * (PART_ROWS  - 1));
            int is_left = (i % (GRID_COLS / WIDTH_FACTOR) == 0) && (k == 0);

            unsigned int idx_k = k*32;
            //c[k] = imgvf_rf[GRID_COLS * (0) + 0 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_c = imgvf_rf[GRID_COLS/WIDTH_FACTOR].range(idx_k + 31, idx_k);
            c[k] = *((float*)(&temp_c));

            //ul[k] = (is_top || is_left) ? c[k] : imgvf_rf[GRID_COLS * (-1) + -1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_ul = (k==0)? imgvf_rf_poped1 : imgvf_rf[0].range(idx_k - 1, idx_k - 32);
            ul[k] = (is_top || is_left) ? c[k] : *((float*)(&temp_ul));

            //u[k] = (is_top) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 0 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_u = imgvf_rf[0].range(idx_k + 31, idx_k);
            u[k] = (is_top)? c[k] : *((float*)(&temp_u));

            //ur[k] = (is_top || is_right) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_ur = (k==15)? imgvf_rf_poped2 : imgvf_rf[0].range(idx_k + 63, idx_k + 32);
            ur[k] = (is_top || is_right)? c[k] : *((float*)(&temp_ur));

            //l[k] = (is_left) ? c[k] : imgvf_rf[GRID_COLS * (0) + -1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_l = (k==0)? imgvf_rf_poped3 : imgvf_rf[GRID_COLS/WIDTH_FACTOR].range(idx_k - 1, idx_k - 32);
            l[k] = (is_left)? c[k] : *((float*)(&temp_l));
    
            //r[k] = (is_right) ? c[k] : imgvf_rf[GRID_COLS * (0) + 1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_r = (k==15)? imgvf_rf_poped4 : imgvf_rf[GRID_COLS/WIDTH_FACTOR].range(idx_k + 63, idx_k + 32);
            r[k] = (is_right)? c[k] : *((float*)(&temp_r));

            //dl[k] = (is_bottom || is_left) ? c[k] : imgvf_rf[GRID_COLS * (1) + -1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_dl = (k==0)? imgvf_rf_poped5 : imgvf_rf[GRID_COLS/WIDTH_FACTOR * 2].range(idx_k - 1, idx_k - 32);
            dl[k] = (is_bottom || is_left) ? c[k] : *((float*)(&temp_dl));

            //d[k] = (is_bottom) ? c[k] : imgvf_rf[GRID_COLS * (1) + 0 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_d = imgvf_rf[GRID_COLS/WIDTH_FACTOR * 2].range(idx_k + 31, idx_k);
            d[k] = (is_bottom) ? c[k] : *((float*)(&temp_d));

            
            //dr[k] = (is_bottom || is_right) ? c[k] : imgvf_rf[GRID_COLS * (1) + 1 + k + GRID_COLS + MAX_RADIUS];
            uint32_t temp_dr = (k==15)? imgvf_rf_poped6 : imgvf_rf[GRID_COLS/WIDTH_FACTOR * 2].range(idx_k + 63, idx_k + 32);
            dr[k] = (is_bottom || is_right) ? c[k] : *((float*)(&temp_dr));

            unsigned int idx = (i*WIDTH_FACTOR+k) / WIDTH_FACTOR;
            unsigned int range_idx =  (i*WIDTH_FACTOR+k) % WIDTH_FACTOR * 32;

            uint32_t temp_I = I[idx].range(range_idx+31, range_idx);
            float read_I = *((float*)(&temp_I));
            vI[k] = read_I;

            float res = lc_mgvf_stencil_core(c[k], ul[k], u[k], ur[k], l[k], r[k], dl[k], d[k], dr[k], vI[k]);
            //res = c[k];
            //res = is_right + 10*is_left + 100*is_top + 1000*is_bottom;
            result[idx].range(range_idx+31, range_idx) = *((uint32_t *)(&res));
        }

        for(;k < WIDTH_FACTOR; k++){
#pragma hls unroll            
            unsigned int idx = (i*WIDTH_FACTOR+k) / WIDTH_FACTOR;
            unsigned int range_idx =  (i*WIDTH_FACTOR+k) % WIDTH_FACTOR * 32;

            float res = 0;
            result[idx].range(range_idx+31, range_idx) = *((uint32_t *)(&res));            
        }

        imgvf_rf_poped1 = imgvf_rf[0].range(511, 480);


DOWN_SHIFT_LOOP:
        for (k = 0; k < GRID_COLS/WIDTH_FACTOR*2 + 1; k++) {
#pragma HLS unroll
            imgvf_rf[k] = imgvf_rf[k + 1];
        }

/*
DOWN_FEED_LOOP:
		for (k = 0; k < PARA_FACTOR; k += 1) {
#pragma HLS unroll    

            	unsigned int idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k)/WIDTH_FACTOR;
                unsigned int range_idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k)%WIDTH_FACTOR*32;
                uint32_t temp_imgvf = imgvf[idx].range(range_idx+31, range_idx);
                float read_imgvf = *((float*)(&temp_imgvf));
            	imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + k] = read_imgvf;
        }
*/

        unsigned int idx = GRID_COLS/WIDTH_FACTOR * 2 + (i + 2);
        imgvf_rf[GRID_COLS/WIDTH_FACTOR*2 + 1] = imgvf[idx];
        
	}

    return;
}

extern "C" {
__kernel void down_kernel(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I, unsigned int bonus, unsigned iter)
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
        lc_mgvf(result, imgvf - GRID_COLS/WIDTH_FACTOR, I, bonus);
        lc_mgvf(imgvf, result - GRID_COLS/WIDTH_FACTOR, I, bonus);
    }

    return;
}
}