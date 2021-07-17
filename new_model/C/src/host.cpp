#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <math.h>

// For LC_MGVF Constants
#include "lc_mgvf.h"

////////////////////RESET FUNCTION//////////////////////////////////
int fill_buffer(float* buffer, std::ifstream &data_input, size_t offset, size_t length) {

    data_input.seekg(std::ios::beg);

    float temp;
    for(size_t i = 0; i < offset; i++){
        data_input >> temp;
    }

    for(size_t i = 0; i < length; i++){
        data_input >> buffer[i];
    }

    return 0;
}

int read_multi_kernel(std::vector<std::vector<float> > &MGVF_buffer,
        std::vector<std::vector<float> > &I_buffer, float* check_val){

    const std::string MGVF_path("../data/mgvf.data");
    const std::string I_path("../data/i.data");
    const std::string check_path("../data/check.data");

    std::ifstream MGVF_file(MGVF_path);
    std::ifstream I_file(I_path);
    std::ifstream check_file(check_path);

#if KERNEL_COUNT==1
    fill_buffer(MGVF_buffer[0].data() + GRID_COLS, MGVF_file, 0, GRID_COLS * PART_ROWS);
    fill_buffer(I_buffer[0].data(), I_file, 0, GRID_COLS * PART_ROWS);
#else
    for(int i = 0; i < KERNEL_COUNT; i++){
        std::cout <<"Point: Load MGVF, iter=" << i << std::endl;
        if(i == 0){
            fill_buffer(MGVF_buffer[0].data() + GRID_COLS, MGVF_file, 0, GRID_COLS * PART_ROWS + GRID_COLS);
        }else if(i == KERNEL_COUNT - 1){
            fill_buffer(MGVF_buffer[i].data(), MGVF_file, GRID_COLS * PART_ROWS * i - GRID_COLS, GRID_COLS * PART_ROWS + GRID_COLS);
        }else{
            fill_buffer(MGVF_buffer[i].data(), MGVF_file, GRID_COLS * PART_ROWS * i - GRID_COLS, GRID_COLS * PART_ROWS + 2 * GRID_COLS);
        }

        fill_buffer(I_buffer[i].data(), I_file, GRID_COLS * PART_ROWS * i, GRID_COLS * PART_ROWS);
    }
#endif

    fill_buffer(check_val, check_file, 0, GRID_COLS * GRID_ROWS);

    MGVF_file.close();
    I_file.close();
    check_file.close();
    return 0;
}

///////////////////VERIFY FUNCTION///////////////////////////////////
bool verify(std::vector<std::vector<float> > results, std::vector<float> check_results) {
    bool match = true;
    std::ofstream out("output.data");
    std::ofstream rpt("report.report");
    out.precision(18);
    out << std::fixed;
    rpt.precision(18);
    rpt <<std::fixed;
    std::cout.precision(18);
    std::cout <<std::fixed;

    for(int i = 0; i < KERNEL_COUNT; i++){
        for(int j = 0; j < GRID_COLS * PART_ROWS; j++){
            out << results[i][j + GRID_COLS] << std::endl;
            if(fabs(results[i][j + GRID_COLS] - check_results[i * GRID_COLS * PART_ROWS + j]) > 1e-17){
                //std::cout << "Unmatch in position" << i << " : " << j / GRID_COLS << " : " << j %GRID_COLS <<
                //    "where " << results[i][i] << " != " << check_results[i * GRID_COLS * PART_ROWS + j];
                rpt << "Unmatch in position" << i << " : " << j / GRID_COLS << " : " << j % GRID_COLS <<
                    "where " << results[i][j + GRID_COLS] << " != " << check_results[i * GRID_COLS * PART_ROWS + j] << std::endl;
                match = false;    
            }
        }
    }
    
    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return match;
}
////////MAIN FUNCTION//////////
int main(int argc, char** argv) {

    unsigned int MGVF_buffer_size = GRID_COLS * PART_ROWS + 2 * GRID_COLS;
    unsigned int I_buffer_size = GRID_COLS * PART_ROWS;
    // I/O Data Vectors
    std::vector<std::vector<float> > MGVFs;
    std::vector<std::vector<float> > Is;
    std::vector<std::vector<float> > results;
    std::vector<float> check_val(GRID_COLS * GRID_ROWS);

    for(int i = 0; i< KERNEL_COUNT; i++){
        MGVFs.emplace_back(MGVF_buffer_size, 0);
        Is.emplace_back(I_buffer_size, 0);
        results.emplace_back(MGVF_buffer_size, 0);
    }

    // Feed input
    read_multi_kernel(MGVFs, Is, check_val.data());

    std::cout << "Input Initialized." << std::endl;

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication

    for(int i = 0; i< KERNEL_COUNT; i++){
        ap_uint<512> result_buffer[(GRID_COLS*PART_ROWS+2*GRID_COLS)/WIDTH_FACTOR], MGVF_buffer[(GRID_COLS*PART_ROWS+2*GRID_COLS)/WIDTH_FACTOR], I_buffer[(GRID_COLS*PART_ROWS)/WIDTH_FACTOR];
        for(int j=0; j<GRID_COLS*PART_ROWS+2*GRID_COLS; j++){
            float temp = MGVFs[i][j];
            MGVF_buffer[j/WIDTH_FACTOR].range(j%WIDTH_FACTOR*32 + 31, j%WIDTH_FACTOR*32) = *((uint32_t *)(&temp));
        }
        for(int j=0; j<GRID_COLS*PART_ROWS; j++){
            float temp = Is[i][j];
            I_buffer[j/WIDTH_FACTOR].range(j%WIDTH_FACTOR*32 + 31, j%WIDTH_FACTOR*32) = *((uint32_t *)(&temp));
        }

        workload(result_buffer, MGVF_buffer, I_buffer, 3, 2);

        for(int j=0; j<GRID_COLS*PART_ROWS+2*GRID_COLS; j++){
            uint32_t temp = (result_buffer[j/WIDTH_FACTOR].range(j%WIDTH_FACTOR*32 + 31, j%WIDTH_FACTOR*32));
            results[i][j] = *((float*)(&temp));
        }
    }
    
    // Compare the device results with software results
    bool match = verify(results, check_val);

    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}