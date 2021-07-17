#include"lc_mgvf.h"

extern "C" {

float heaviside(float x) {
    // A simpler, faster approximation of the Heaviside function
    float out = 0.0;
    if (x > -0.0001) out = 0.5;
    if (x >  0.0001) out = 1.0;
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


float lc_mgvf(float result[GRID_ROWS * GRID_COLS], float imgvf[GRID_ROWS * GRID_COLS], float I[GRID_ROWS * GRID_COLS])
{
    float total_diff = 0.0;

    for (int i = 0; i < GRID_ROWS; i++) {
        for (int j = 0; j < GRID_COLS; j++) {
            /*
            float old_val = imgvf[i * GRID_COLS + j];

            float UL    = ((i == 0              ||  j == 0              ) ? 0 : (imgvf[(i - 1   ) * GRID_COLS + (j - 1  )] - old_val));
            float U     = ((i == 0                                      ) ? 0 : (imgvf[(i - 1   ) * GRID_COLS + (j      )] - old_val));
            float UR    = ((i == 0              ||  j == GRID_COLS - 1  ) ? 0 : (imgvf[(i - 1   ) * GRID_COLS + (j + 1  )] - old_val));

            float L     = ((                        j == 0              ) ? 0 : (imgvf[(i       ) * GRID_COLS + (j - 1  )] - old_val));
            float R     = ((                        j == GRID_COLS - 1  ) ? 0 : (imgvf[(i       ) * GRID_COLS + (j + 1  )] - old_val));

            float DL    = ((i == GRID_ROWS - 1  ||  j == 0              ) ? 0 : (imgvf[(i + 1   ) * GRID_COLS + (j - 1  )] - old_val));
            float D     = ((i == GRID_ROWS - 1                          ) ? 0 : (imgvf[(i + 1   ) * GRID_COLS + (j      )] - old_val));
            float DR    = ((i == GRID_ROWS - 1  ||  j == GRID_COLS - 1  ) ? 0 : (imgvf[(i + 1   ) * GRID_COLS + (j + 1  )] - old_val));

            float vHe = old_val + MU_O_LAMBDA * (heaviside(UL) * UL + heaviside(U) * U + heaviside(UR) * UR + heaviside(L) * L + heaviside(R) * R + heaviside(DL) * DL + heaviside(D) * D + heaviside(DR) * DR);

            float vI = I[i * GRID_COLS + j];
            float new_val = vHe - (ONE_O_LAMBDA * vI * (vHe - vI));
            result[i * GRID_COLS + j] = new_val;

            total_diff += fabs(new_val - old_val);
            */
            float c = imgvf[i * GRID_COLS + j];

            float ul    = ((i == 0              ||  j == 0              ) ? c : (imgvf[(i - 1   ) * GRID_COLS + (j - 1  )]));
            float u     = ((i == 0                                      ) ? c : (imgvf[(i - 1   ) * GRID_COLS + (j      )]));
            float ur    = ((i == 0              ||  j == GRID_COLS - 1  ) ? c : (imgvf[(i - 1   ) * GRID_COLS + (j + 1  )]));

            float l     = ((                        j == 0              ) ? c : (imgvf[(i       ) * GRID_COLS + (j - 1  )]));
            float r     = ((                        j == GRID_COLS - 1  ) ? c : (imgvf[(i       ) * GRID_COLS + (j + 1  )]));

            float dl    = ((i == GRID_ROWS - 1  ||  j == 0              ) ? c : (imgvf[(i + 1   ) * GRID_COLS + (j - 1  )]));
            float d     = ((i == GRID_ROWS - 1                          ) ? c : (imgvf[(i + 1   ) * GRID_COLS + (j      )]));
            float dr    = ((i == GRID_ROWS - 1  ||  j == GRID_COLS - 1  ) ? c : (imgvf[(i + 1   ) * GRID_COLS + (j + 1  )]));

            float vI = I[i * GRID_COLS + j];

            float res = lc_mgvf_stencil_core(c, ul, u, ur, l, r, dl, d, dr, vI);

            result[i * GRID_COLS + j] = res;

            total_diff += fabs(res - c);
        }
    }

    return (total_diff / (float)(GRID_ROWS * GRID_COLS));
}

void workload(float result[GRID_ROWS * GRID_COLS], float imgvf[GRID_ROWS * GRID_COLS], float I[GRID_ROWS * GRID_COLS])
{
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result1
    #pragma HLS INTERFACE m_axi port=imgvf offset=slave bundle=imgvf1
    #pragma HLS INTERFACE m_axi port=I offset=slave bundle=I1
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=imgvf bundle=control
    #pragma HLS INTERFACE s_axilite port=I bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int i;
    float diff = 1.0;
    for (i = 0; i < ITERATION / 2; i++) {
        diff = lc_mgvf(result, imgvf, I);
        diff = lc_mgvf(imgvf, result, I);
    }
    return;

}








}
