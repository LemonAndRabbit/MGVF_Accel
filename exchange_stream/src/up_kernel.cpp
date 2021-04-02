#include "lc_mgvf.h"

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

typedef ap_axiu<DWIDTH, 0, 0, 0> pkt;

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
		for (k = 0; k < PARA_FACTOR; k += WIDTH_FACTOR) {
#pragma HLS pipeline II=1
            for(int g = 0; g < WIDTH_FACTOR; g++){
#pragma HLS unroll
                unsigned int idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k + g)/WIDTH_FACTOR;
                if(idx < GRID_COLS*(PART_ROWS+1)/WIDTH_FACTOR || first_round){
                    //unsigned int idx = (GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k + g)/WIDTH_FACTOR;
                    //unsigned int range_idx = (GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k + g)%WIDTH_FACTOR*32;
                    unsigned int range_idx = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR + k + g)%WIDTH_FACTOR*32;
                    uint32_t temp_imgvf = imgvf[idx].range(range_idx+31, range_idx);
                    float read_imgvf = *((float*)(&temp_imgvf));
                    //imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + g + k] = read_imgvf;
                    imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + g + k] = read_imgvf;
                }
                else{
                    unsigned int idx_2 = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR - GRID_COLS * (PART_ROWS+1) + k + g)/WIDTH_FACTOR;
                    unsigned int range_idx_2 = (GRID_COLS * (2 * MAX_RADIUS) + PARA_FACTOR + (i + 1) * PARA_FACTOR - GRID_COLS * (PART_ROWS+1) + k + g)%WIDTH_FACTOR*32;
                    uint32_t temp_imgvf_2 = buffer_from[idx_2].range(range_idx_2+31, range_idx_2);
                    float read_imgvf_2 = *((float*)(&temp_imgvf_2));
                    imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR + g + k] = read_imgvf_2;
                }
            }
        }
    }

	return;
}

static void send_data(INTERFACE_WIDTH *buffer_to, hls::stream<pkt> &port_to){
    int i;
    for(i = 0; i < GRID_COLS/WIDTH_FACTOR; i++){
    #pragma HLS pipeline II=1    
        pkt temp;
        temp.data = buffer_to[i];
        port_to.write(temp);
    }
}

static void receive_data(INTERFACE_WIDTH *buffer_from, hls::stream<pkt> &port_from){
    int i;
    for(i = 0; i < GRID_COLS/WIDTH_FACTOR; i++){
    #pragma HLS pipeline II=1    
        pkt temp = port_from.read();
        buffer_from[i] = temp.data;
    }
}


static void data_exchange(INTERFACE_WIDTH *buffer_to, INTERFACE_WIDTH *buffer_from, hls::stream<pkt> &port_to, hls::stream<pkt> &port_from){
    send_data(buffer_to, port_to);
    receive_data(buffer_from, port_from);
}

extern "C" {
__kernel void up_kernel(INTERFACE_WIDTH *result, INTERFACE_WIDTH *imgvf, INTERFACE_WIDTH *I, hls::stream<pkt> &port_to, hls::stream<pkt> &port_from)
{
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result1
    #pragma HLS INTERFACE m_axi port=imgvf offset=slave bundle=imgvf1
    #pragma HLS INTERFACE m_axi port=I offset=slave bundle=I1
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=imgvf bundle=control
    #pragma HLS INTERFACE s_axilite port=I bundle=control

    #pragma HLS INTERFACE axis port=port_to bundle=port_to
    #pragma HLS INTERFACE axis port=port_from bundle=port_from
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    INTERFACE_WIDTH buffer_to[GRID_COLS/WIDTH_FACTOR];
    INTERFACE_WIDTH buffer_from[GRID_COLS/WIDTH_FACTOR];

    int i;
    for(i=0; i<ITERATION/2; i++){
        lc_mgvf(result, imgvf - GRID_COLS/WIDTH_FACTOR, I, buffer_to, buffer_from, (i==0));
        data_exchange(buffer_to, buffer_from, port_to, port_from);
        lc_mgvf(imgvf, result - GRID_COLS/WIDTH_FACTOR, I, buffer_to, buffer_from, false);
        data_exchange(buffer_to, buffer_from, port_to, port_from);
    }
    return;
}
}