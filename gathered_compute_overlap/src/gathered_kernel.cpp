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

static void down_lc_mgvf(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I)
{
    int cols = GRID_COLS;
    int rows = PART_ROWS;
    float imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + PARA_FACTOR];
#pragma HLS array_partition variable=imgvf_rf complete dim=0

    int i;

    int input_bound = GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + PARA_FACTOR;
DOWN_INITIALZE_LOOP:    
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
DOWN_MAJOR_LOOP:
    for (i = 0; i < GRID_COLS / PARA_FACTOR * (PART_ROWS+ITERATION); i++) {
        int k;
#pragma HLS pipeline II=1
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
            int is_bottom = (i >= GRID_COLS / PARA_FACTOR * (PART_ROWS + ITERATION  - 1));
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
            //float res = c[k];
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

                //unsigned int idx = (GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k)/WIDTH_FACTOR;
            	unsigned int idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k)/WIDTH_FACTOR;
                unsigned int range_idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k)%WIDTH_FACTOR*32;
                uint32_t temp_imgvf = imgvf[idx].range(range_idx+31, range_idx);
                float read_imgvf = *((float*)(&temp_imgvf));
                //imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] = read_imgvf;
            	imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + k] = read_imgvf;
        }
	}


    return;
}

void up_lc_mgvf(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I)
{
    int cols = GRID_COLS;
    int rows = PART_ROWS;
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
    for (i = 0; i < GRID_COLS / PARA_FACTOR * (PART_ROWS+ITERATION); i++) {
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
            //float res = c[k];
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


static void down_kernel(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I, unsigned iter)
{

    int i;
    for(i=0; i<iter/2; i++){
        down_lc_mgvf(result, imgvf - GRID_COLS/WIDTH_FACTOR, I);
        down_lc_mgvf(imgvf, result - GRID_COLS/WIDTH_FACTOR, I);
    }

    return;
}

static void up_kernel(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I, unsigned iter)
{
    int i;
    for(i=0; i<iter/2; i++){
        up_lc_mgvf(result, imgvf - GRID_COLS/WIDTH_FACTOR, I);
        up_lc_mgvf(imgvf, result - GRID_COLS/WIDTH_FACTOR, I);
    }

    return;
}

extern "C"{
__kernel void gathered_kernel(INTERFACE_WIDTH *up_result, INTERFACE_WIDTH *up_imgvf, INTERFACE_WIDTH *up_I, 
    INTERFACE_WIDTH *down_result, INTERFACE_WIDTH *down_imgvf, INTERFACE_WIDTH *down_I, unsigned iter){
#pragma HLS INTERFACE m_axi port=up_result offset=slave bundle=result1
#pragma HLS INTERFACE m_axi port=up_imgvf offset=slave bundle=imgvf1
#pragma HLS INTERFACE m_axi port=up_I offset=slave bundle=I1

#pragma HLS INTERFACE m_axi port=down_result offset=slave bundle=result2
#pragma HLS INTERFACE m_axi port=down_imgvf offset=slave bundle=imgvf2
#pragma HLS INTERFACE m_axi port=down_I offset=slave bundle=I2

#pragma HLS INTERFACE s_axilite port=up_result
#pragma HLS INTERFACE s_axilite port=up_imgvf
#pragma HLS INTERFACE s_axilite port=up_I

#pragma HLS INTERFACE s_axilite port=down_result
#pragma HLS INTERFACE s_axilite port=down_imgvf
#pragma HLS INTERFACE s_axilite port=down_I

#pragma HLS INTERFACE s_axilite port=return    
    
    up_kernel(up_result, up_imgvf, up_I, iter);
    down_kernel(down_result, down_imgvf, down_I, iter);
}    
}