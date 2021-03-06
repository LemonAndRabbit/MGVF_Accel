#include "lc_mgvf.h"

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

typedef ap_axiu<1, 0, 0, 0> pkt;

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

static void lc_mgvf(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I, INTERFACE_WIDTH *buffer_to, INTERFACE_WIDTH *buffer_from, bool first_round)
{
	int cols = GRID_COLS;
	int rows = PART_ROWS;
	float imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + PARA_FACTOR];
#pragma HLS array_partition variable=imgvf_rf complete dim=0

	int i;

INITIALIZE_LOOP:
    int input_bound = GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + PARA_FACTOR;
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

MAJOR_LOOP:
    for (i = 0; i < GRID_COLS / PARA_FACTOR * PART_ROWS; i++) {
		int k;
#pragma HLS pipeline II=1

        int is_bottom = (i >= GRID_COLS / PARA_FACTOR * (PART_ROWS - 1));
        INTERFACE_WIDTH temp512;
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
			int is_top = (i < GRID_COLS / PARA_FACTOR);
			int is_right = (i % (GRID_COLS / PARA_FACTOR) == (GRID_COLS / PARA_FACTOR - 1)) && (k == PARA_FACTOR - 1);
			int is_left = (i % (GRID_COLS / PARA_FACTOR) == 0) && (k == 0);

			c[k] = imgvf_rf[GRID_COLS * (0) + 0 + k + GRID_COLS + MAX_RADIUS];
			ul[k] = (is_top || is_left) ? c[k] : imgvf_rf[GRID_COLS * (-1) + -1 + k + GRID_COLS + MAX_RADIUS];
			u[k] = (is_top) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 0 + k + GRID_COLS + MAX_RADIUS];
			ur[k] = (is_top || is_right) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 1 + k + GRID_COLS + MAX_RADIUS];
			l[k] = (is_left) ? c[k] : imgvf_rf[GRID_COLS * (0) + -1 + k + GRID_COLS + MAX_RADIUS];
			r[k] = (is_right) ? c[k] : imgvf_rf[GRID_COLS * (0) + 1 + k + GRID_COLS + MAX_RADIUS];
			dl[k] = (is_left) ? c[k] : imgvf_rf[GRID_COLS * (1) + -1 + k + GRID_COLS + MAX_RADIUS];
			d[k] = imgvf_rf[GRID_COLS * (1) + 0 + k + GRID_COLS + MAX_RADIUS];
			dr[k] = (is_right) ? c[k] : imgvf_rf[GRID_COLS * (1) + 1 + k + GRID_COLS + MAX_RADIUS];

            unsigned int idx = (i*PARA_FACTOR+k) / WIDTH_FACTOR;
            unsigned int range_idx =  (i*PARA_FACTOR+k) % WIDTH_FACTOR * 32;

            uint32_t temp_I = I[idx].range(range_idx+31, range_idx);
            float read_I = *((float*)(&temp_I));
            vI[k] = read_I;

            float res = lc_mgvf_stencil_core(c[k], ul[k], u[k], ur[k], l[k], r[k], dl[k], d[k], dr[k], vI[k]);
            //float res = d[k];
			result[idx].range(range_idx+31, range_idx) = *((uint32_t *)(&res));
            /*
            if(is_bottom){
                unsigned int dk_idx = (i * PARA_FACTOR + k - GRID_COLS * (PART_ROWS - 1)) / WIDTH_FACTOR;
                unsigned int dk_range_idx =  (i * PARA_FACTOR + k - GRID_COLS * (PART_ROWS - 1)) % WIDTH_FACTOR * 32;
                buffer_to[dk_idx].range(dk_range_idx+31, dk_range_idx) = *((uint32_t *)(&res));
            }
            */
           temp512.range(range_idx+31, range_idx) = *((uint32_t *)(&res));
		}

        if(is_bottom){
            unsigned int dk_idx = (i * PARA_FACTOR - GRID_COLS * (PART_ROWS - 1)) / WIDTH_FACTOR;
            buffer_to[dk_idx] = temp512;
        }
SHIFT_LOOP:
		for (k = 0; k < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; k++) {
#pragma HLS unroll
			imgvf_rf[k] = imgvf_rf[k + PARA_FACTOR];
		}

FEED_LOOP:
        unsigned int idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR)/WIDTH_FACTOR;
        unsigned int idx2 = idx%(GRID_COLS/WIDTH_FACTOR);
        INTERFACE_WIDTH temp_imgvf1 = imgvf[idx];
        INTERFACE_WIDTH temp_imgvf2 = buffer_from[idx2];
        INTERFACE_WIDTH temp_imgvf3 = 
            (idx < GRID_COLS*(PART_ROWS+1)/WIDTH_FACTOR || first_round)?temp_imgvf1:temp_imgvf2;
        for(int g = 0; g < WIDTH_FACTOR; g++){
#pragma HLS unroll
            unsigned int range_idx = g*32;
            uint32_t temp_imgvf4 = temp_imgvf3.range(range_idx+31, range_idx);
            float read_imgvf = *((float*)(&temp_imgvf4));
            imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + g] = read_imgvf;
        }   
    }  

	return;
}

static void make_syn(hls::stream<pkt> &syn_master){
    pkt temp;
    syn_master.write(temp);
}

extern "C" {
__kernel void up_kernel(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I, INTERFACE_WIDTH *port_to, INTERFACE_WIDTH *port_from, hls::stream<pkt> &syn_master)
{
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result1
    #pragma HLS INTERFACE m_axi port=imgvf offset=slave bundle=imgvf1
    #pragma HLS INTERFACE m_axi port=I offset=slave bundle=I1
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=imgvf bundle=control
    #pragma HLS INTERFACE s_axilite port=I bundle=control

    #pragma HLS INTERFACE m_axi port=port_to bundle=port_to
    #pragma HLS INTERFACE m_axi port=port_from bundle=port_from

    #pragma HLS INTERFACE s_axilite port=port_to bundle=control
    #pragma HLS INTERFACE s_axilite port=port_from bundle=control

    #pragma HLS INTERFACE axis port=syn_master
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control



    int i;
    for(i=0; i<ITERATION*2; i++){
        if(i%4==0){
            lc_mgvf(result, imgvf - GRID_COLS/WIDTH_FACTOR, I, port_to, port_from + GRID_COLS, (i==0));
        }
        else if(i%4==1) {
            make_syn(syn_master);  
        }
        else if(i%4==2) {
            lc_mgvf(imgvf, result - GRID_COLS/WIDTH_FACTOR, I, port_to + GRID_COLS, port_from, false);
        }     
        else if(i%4==3){
            make_syn(syn_master);  
        } 
    }
    return;
}
}