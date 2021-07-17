#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <math.h>

// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "xcl2.hpp"
// For LC_MGVF Constants
#include "lc_mgvf.h"


// HBM Pseudo-channel(PC) requirements
#define MAX_HBM_PC_COUNT 32
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
    PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15),
    PC_NAME(16), PC_NAME(17), PC_NAME(18), PC_NAME(19), PC_NAME(20), PC_NAME(21), PC_NAME(22), PC_NAME(23),
    PC_NAME(24), PC_NAME(25), PC_NAME(26), PC_NAME(27), PC_NAME(28), PC_NAME(29), PC_NAME(30), PC_NAME(31)};


auto constexpr test_size = GRID_ROWS * GRID_COLS;
auto constexpr top_size = test_size/2 + ITERATION*GRID_COLS;

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

int read_multi_kernel(std::vector<std::vector<float, aligned_allocator<float> > > &MGVF_buffer,
        std::vector<std::vector<float, aligned_allocator<float> > > &I_buffer, float* check_val){

    const std::string MGVF_path("../data/mgvf.data");
    const std::string I_path("../data/i.data");
    const std::string check_path("../data/check_op1.data");

    std::ifstream MGVF_file(MGVF_path);
    std::ifstream I_file(I_path);
    std::ifstream check_file(check_path);

#if KERNEL_COUNT==1
    fill_buffer(MGVF_buffer[0].data() + GRID_COLS, MGVF_file, 0, GRID_COLS * PART_ROWS);
    fill_buffer(I_buffer[0].data(), I_file, GRID_COLS * PART_ROWS * i, GRID_COLS * PART_ROWS);
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
bool verify(std::vector<std::vector<float, aligned_allocator<float> > > results, std::vector<float> check_results) {
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
    std::vector<std::vector<float, aligned_allocator<float> > > MGVFs;
    std::vector<std::vector<float, aligned_allocator<float> > > Is;
    std::vector<std::vector<float, aligned_allocator<float> > > results;
    std::vector<float> check_val(GRID_COLS * GRID_ROWS);

    for(int i = 0; i< KERNEL_COUNT; i++){
        MGVFs.emplace_back(MGVF_buffer_size, 0);
        Is.emplace_back(I_buffer_size, 0);
        results.emplace_back(MGVF_buffer_size, 0);
    }

/*
    std::vector<float, aligned_allocator<float> > up_MGVF(top_size,0);
    std::vector<float, aligned_allocator<float> > up_I(top_size,0);
    std::vector<float, aligned_allocator<float> > down_MGVF(top_size,0);
    std::vector<float, aligned_allocator<float> > down_I(top_size,0);
    std::vector<float, aligned_allocator<float> > up_results(top_size,0);
    std::vector<float, aligned_allocator<float> > down_results(top_size,0);
    std::vector<float> check_val(GRID_COLS * GRID_ROWS);

    MGVFs.push_back(up_MGVF.data());
    MGVFs.push_back(down_MGVF.data());
    Is.push_back(up_I.data());
    Is.push_back(down_I.data());
    results.push_back(up_results.data());
    results.push_back(down_results.data());
*/

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    // OpenCL Host Code Begins.
    cl_int err;

    // OpenCL objects
    cl::Device device;
    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;
    std::vector<cl::Kernel> kernels;

    auto binaryFile = argv[1];

    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                            CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            // Creating Kernel
            char kernel_name[50];
            for(int k = 0; k < KERNEL_COUNT; k++){
                sprintf(kernel_name, "unikernel:{unikernel_%d}", k+1);
                OCL_CHECK(err, kernels.emplace_back(program, kernel_name, &err));
            }
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "MGVF Kernel Loaded." <<std::endl;

    // Feed input
    read_multi_kernel(MGVFs, Is, check_val.data());

    std::cout << "Input Initialized." << std::endl;

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication

    std::vector<cl_mem_ext_ptr_t> ptr_MGVF(KERNEL_COUNT), ptr_I(KERNEL_COUNT), ptr_results(KERNEL_COUNT);

    // For Allocating Buffer to specific Global Memory PC, user has to use
    // cl_mem_ext_ptr_t
    // and provide the PCs

    for(int i = 0; i < KERNEL_COUNT; i++){
        ptr_MGVF[i].obj = MGVFs[i].data();
        ptr_I[i].obj = Is[i].data();
        ptr_results[i].obj = results[i].data();

        ptr_MGVF[i].param = 0;
        ptr_I[i].param = 0;
        ptr_results[i].param = 0;

        ptr_MGVF[i].flags = pc[16*i+1];
        ptr_I[i].flags = pc[16*i+2];
        ptr_results[i].flags = pc[16*i+3];
    }

    std::vector<cl::Buffer> device_MGVF, device_I, device_results;

    for(int i = 0; i < KERNEL_COUNT; i++){
        OCL_CHECK(err, device_MGVF.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, MGVF_buffer_size*sizeof(float),
            &ptr_MGVF[i], &err));
        OCL_CHECK(err, device_I.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, I_buffer_size*sizeof(float),
            &ptr_I[i], &err));
        OCL_CHECK(err, device_results.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, MGVF_buffer_size*sizeof(float),
            &ptr_results[i], &err));
    }

    // Setting Kernel Arguments
    // and Copy input data to device global memory
    printf("Starting write device buffer!\n");

    for(int i = 0; i < KERNEL_COUNT; i++){
        OCL_CHECK(err, err = kernels[i].setArg(0, device_results[i]));
        OCL_CHECK(err, err = kernels[i].setArg(1, device_MGVF[i]));
        OCL_CHECK(err, err = kernels[i].setArg(2, device_I[i]));
        OCL_CHECK(err, err = kernels[i].setArg(3, i==0?1:(i==KERNEL_COUNT-1?2:4)));
        OCL_CHECK(err, err = kernels[i].setArg(4, ITERATION));
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({device_MGVF[i], device_I[i]}, 0 /* 0 means from host*/));
    }

    q.finish();

    printf("Write device buffer finished!\n");

    // Launch the Kernel
    for(int i = 0; i< KERNEL_COUNT; i++){
        OCL_CHECK(err, err = q.enqueueTask(kernels[i]));
    }
    q.finish();

    // Copy Result from Device Global Memory to Host Local Memory
    
    printf("Execution finished.\n");
    printf("Strating read device buffer!\n");
    
    for(int i = 0; i< KERNEL_COUNT; i++){
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({device_results[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
    }
    q.finish();
    // OpenCL Host Code Ends
    printf("Read result finished!\n");

    // Compare the device results with software results
    bool match = verify(results, check_val);

    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}