#include "lc_mgvf.h"

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

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

static float lc_mgvf_stencil_check_core(float c, float ul, float u, float ur, float l, float r, float dl, float d, float dr, float vI)
{  
    return c + ul + u + ur + l + r + dl + d + dr;
}


static void down_lc_mgvf(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I)
{
    float imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + PARA_FACTOR];
#pragma HLS array_partition variable=imgvf_rf complete dim=0

    int i;

    int input_bound = GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + PARA_FACTOR;
DOWN_INITIALZE_LOOP:    
    for (i = 0; i < input_bound; i+= WIDTH_FACTOR) {
#pragma HLS pipeline II=1
        for (int j = 0; j < WIDTH_FACTOR; j++){
#pragma HLS unroll
            unsigned int range_idx = j*32;
            uint32_t temp_imgvf = imgvf[i/WIDTH_FACTOR].range(range_idx+31, range_idx);
            float read_imgvf = *((float*)(&temp_imgvf));
            imgvf_rf[i + j + MAX_RADIUS] = read_imgvf;
        }
    }

DOWN_MAJOR_LOOP:
    for (i = 0; i < GRID_COLS / PARA_FACTOR * PART_ROWS; i++) {
#pragma HLS pipeline II=1        
        int k;

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
            int is_right = (i % (GRID_COLS / PARA_FACTOR) == (GRID_COLS / PARA_FACTOR - 1)) && (k == PARA_FACTOR - 1);
            int is_bottom = (i >= GRID_COLS / PARA_FACTOR * (PART_ROWS  - 1));
            int is_left = (i % (GRID_COLS / PARA_FACTOR) == 0) && (k == 0);

            c[k] = imgvf_rf[GRID_COLS * (0) + 0 + k + GRID_COLS + MAX_RADIUS];
            ul[k] = (is_top || is_left) ? c[k] : imgvf_rf[GRID_COLS * (-1) + -1 + k + GRID_COLS + MAX_RADIUS];
            u[k] = (is_top) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 0 + k + GRID_COLS + MAX_RADIUS];
            ur[k] = (is_top || is_right) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 1 + k + GRID_COLS + MAX_RADIUS];
            l[k] = (is_left) ? c[k] : imgvf_rf[GRID_COLS * (0) + -1 + k + GRID_COLS + MAX_RADIUS];
            r[k] = (is_right) ? c[k] : imgvf_rf[GRID_COLS * (0) + 1 + k + GRID_COLS + MAX_RADIUS];
            dl[k] = (is_bottom || is_left) ? c[k] : imgvf_rf[GRID_COLS * (1) + -1 + k + GRID_COLS + MAX_RADIUS];
            d[k] = (is_bottom) ? c[k] : imgvf_rf[GRID_COLS * (1) + 0 + k + GRID_COLS + MAX_RADIUS];
            dr[k] = (is_bottom || is_right) ? c[k] : imgvf_rf[GRID_COLS * (1) + 1 + k + GRID_COLS + MAX_RADIUS];

            unsigned int idx = (i*PARA_FACTOR+k) / WIDTH_FACTOR;
            unsigned int range_idx =  (i*PARA_FACTOR+k) % WIDTH_FACTOR * 32;

            uint32_t temp_I = I[idx].range(range_idx+31, range_idx);
            float read_I = *((float*)(&temp_I));
            vI[k] = read_I;

            float res = lc_mgvf_stencil_core(c[k], ul[k], u[k], ur[k], l[k], r[k], dl[k], d[k], dr[k], vI[k]);
            //res = is_right + 10*is_left + 100*is_top + 1000*is_bottom;
            result[idx].range(range_idx+31, range_idx) = *((uint32_t *)(&res));
        }

DOWN_SHIFT_LOOP:
        for (k = 0; k < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; k++) {
#pragma HLS unroll
            imgvf_rf[k] = imgvf_rf[k + PARA_FACTOR];
        }

DOWN_FEED_LOOP:
		for (k = 0; k < PARA_FACTOR; k += 1) {
#pragma HLS unroll    

            	unsigned int idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k)/WIDTH_FACTOR;
                unsigned int range_idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k)%WIDTH_FACTOR*32;
                uint32_t temp_imgvf = imgvf[idx].range(range_idx+31, range_idx);
                float read_imgvf = *((float*)(&temp_imgvf));
            	imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + k] = read_imgvf;
        }
	}

    return;
}

void up_lc_mgvf(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I)
{
    float imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + PARA_FACTOR];
#pragma HLS array_partition variable=imgvf_rf complete dim=0

    int i;

    int input_bound = GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + PARA_FACTOR;
UP_INITIALIZE_LOOP:    
    for (i = GRID_COLS; i < input_bound; i+= WIDTH_FACTOR) {
#pragma HLS pipeline II=1
        for (int j = 0; j < WIDTH_FACTOR; j++){
#pragma HLS unroll
            unsigned int range_idx = j*32;
            uint32_t temp_imgvf = imgvf[i/WIDTH_FACTOR].range(range_idx+31, range_idx);
            float read_imgvf = *((float*)(&temp_imgvf));
            imgvf_rf[i + j + MAX_RADIUS] = read_imgvf;
        }
    }
UP_MAJOR_LOOP:
    for (i = 0; i < GRID_COLS / PARA_FACTOR * PART_ROWS; i++) {
        int k;
#pragma HLS pipeline II=1
UP_COMPUTE_LOOP:
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
            int is_top = (i < GRID_COLS / PARA_FACTOR);
            int is_right = (i % (GRID_COLS / PARA_FACTOR) == (GRID_COLS / PARA_FACTOR - 1)) && (k == PARA_FACTOR - 1);
            int is_bottom = false;
            int is_left = (i % (GRID_COLS / PARA_FACTOR) == 0) && (k == 0);

            c[k] = imgvf_rf[GRID_COLS * (0) + 0 + k + GRID_COLS + MAX_RADIUS];
            ul[k] = (is_top || is_left) ? c[k] : imgvf_rf[GRID_COLS * (-1) + -1 + k + GRID_COLS + MAX_RADIUS];
            u[k] = (is_top) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 0 + k + GRID_COLS + MAX_RADIUS];
            ur[k] = (is_top || is_right) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 1 + k + GRID_COLS + MAX_RADIUS];
            l[k] = (is_left) ? c[k] : imgvf_rf[GRID_COLS * (0) + -1 + k + GRID_COLS + MAX_RADIUS];
            r[k] = (is_right) ? c[k] : imgvf_rf[GRID_COLS * (0) + 1 + k + GRID_COLS + MAX_RADIUS];
            dl[k] = (is_bottom || is_left) ? c[k] : imgvf_rf[GRID_COLS * (1) + -1 + k + GRID_COLS + MAX_RADIUS];
            d[k] = (is_bottom) ? c[k] : imgvf_rf[GRID_COLS * (1) + 0 + k + GRID_COLS + MAX_RADIUS];
            dr[k] = (is_bottom || is_right) ? c[k] : imgvf_rf[GRID_COLS * (1) + 1 + k + GRID_COLS + MAX_RADIUS];

            unsigned int idx = (i*PARA_FACTOR+k) / WIDTH_FACTOR;
            unsigned int range_idx =  (i*PARA_FACTOR+k) % WIDTH_FACTOR * 32;

            uint32_t temp_I = I[idx].range(range_idx+31, range_idx);
            float read_I = *((float*)(&temp_I));
            vI[k] = read_I;

            float res = lc_mgvf_stencil_core(c[k], ul[k], u[k], ur[k], l[k], r[k], dl[k], d[k], dr[k], vI[k]);
            //res = is_right + 10*is_left + 100*is_top + 1000*is_bottom;
            result[idx].range(range_idx+31, range_idx) = *((uint32_t *)(&res));

        }
UP_SHIFT_LOOP:
        for (k = 0; k < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; k++) {
#pragma HLS unroll
            imgvf_rf[k] = imgvf_rf[k + PARA_FACTOR];
        }
UP_FEED_LOOP:
        for (k = 0; k < PARA_FACTOR; k += WIDTH_FACTOR) {
#pragma HLS pipeline II=1
            for(int g = 0; g < WIDTH_FACTOR; g++){
#pragma HLS unroll
                unsigned int idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k + g)/WIDTH_FACTOR;
                unsigned int range_idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k + g)%WIDTH_FACTOR*32;
                uint32_t temp_imgvf = imgvf[idx].range(range_idx+31, range_idx);
                float read_imgvf = *((float*)(&temp_imgvf));
                imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + g + k] = read_imgvf;
            }
        }

    }

    return;
}

static void exchange_data(INTERFACE_WIDTH *up_result, INTERFACE_WIDTH *down_result){
    int i;
    for(i = 0; i < GRID_COLS / PARA_FACTOR; i++){
        down_result[i - GRID_COLS/PARA_FACTOR] = up_result[i + GRID_COLS/PARA_FACTOR*(PART_ROWS-1)];
    }
    for(i = 0; i < GRID_COLS / PARA_FACTOR; i++){
        up_result[i + GRID_COLS/PARA_FACTOR*PART_ROWS] = down_result[i];
    }
}

extern "C" {
__kernel void gathered_kernel(INTERFACE_WIDTH *up_result, INTERFACE_WIDTH *up_imgvf, INTERFACE_WIDTH *up_I, 
    INTERFACE_WIDTH *down_result, INTERFACE_WIDTH *down_imgvf, INTERFACE_WIDTH *down_I
    /*INTERFACE_WIDTH *buffer1, INTERFACE_WIDTH *buffer2*/){
#pragma HLS INTERFACE m_axi port=up_result offset=slave bundle=result1
#pragma HLS INTERFACE m_axi port=up_imgvf offset=slave bundle=imgvf1
#pragma HLS INTERFACE m_axi port=up_I offset=slave bundle=I1

#pragma HLS INTERFACE m_axi port=down_result offset=slave bundle=re
#pragma HLS INTERFACE m_axi port=down_imgvf offset=slave bundle=imgvf2
#pragma HLS INTERFACE m_axi port=down_I offset=slave bundle=I2

#pragma HLS INTERFACE s_axilite port=up_result
#pragma HLS INTERFACE s_axilite port=up_imgvf
#pragma HLS INTERFACE s_axilite port=up_I

#pragma HLS INTERFACE s_axilite port=down_result
#pragma HLS INTERFACE s_axilite port=down_imgvf
#pragma HLS INTERFACE s_axilite port=down_I

/*
#pragma HLS INTERFACE m_axi port=buffer1 bundle=buffer1_1
#pragma HLS INTERFACE m_axi port=buffer2 bundle=buffer2_1

#pragma HLS INTERFACE s_axilite port=buffer1
#pragma HLS INTERFACE s_axilite port=buffer2
*/
#pragma HLS INTERFACE s_axilite port=return     


    int i;
    down_imgvf = down_imgvf + GRID_COLS/WIDTH_FACTOR;
    down_result = down_result + GRID_COLS/WIDTH_FACTOR;
    //exchange_data(up_imgvf, down_imgvf);
GATHERED_LOOP:    
    for(i=0; i<ITERATION; i++){
        if(i%2==0){
            up_lc_mgvf(up_result, up_imgvf - GRID_COLS/WIDTH_FACTOR, up_I/*, buffer1, buffer2 + GRID_COLS, (i==0)*/);
            down_lc_mgvf(down_result, down_imgvf - GRID_COLS/WIDTH_FACTOR, down_I/*, buffer2, buffer1 + GRID_COLS, (i==0)*/);
            exchange_data(up_result, down_result);
        }    
        else if(i%2==1){
            up_lc_mgvf(up_imgvf, up_result - GRID_COLS/WIDTH_FACTOR, up_I/*, buffer1 + GRID_COLS, buffer2, false*/);
            down_lc_mgvf(down_imgvf, down_result - GRID_COLS/WIDTH_FACTOR, down_I/*, buffer2 + GRID_COLS, buffer1, false*/);
            exchange_data(up_imgvf, down_imgvf);
        }
    }
}    
}