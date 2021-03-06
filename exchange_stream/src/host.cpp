#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <fstream>

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

////////////////////RESET FUNCTION//////////////////////////////////
int reset(float* up_MGVF, float* up_I, float* down_MGVF, float* down_I, float* up_results, float* down_results, float* check_results) {
    /*
    // No care
    std::generate(a, a + size, std::rand);
    std::generate(b, b + size, std::rand);
    std::generate(c, c + size, std::rand);
    for (size_t i = 0; i < size; i++) {
        hw_results[i] = 0;
        sw_results[i] = (a[i] + b[i]) * c[i];
    }
    */
    std::ifstream data_input_I("../data/inputI.data");
    std::ifstream data_input_MGVF("../data/inputMGVF.data");
    std::ifstream check_output("../data/check1.data");
    for(size_t i = 0; i < test_size/2 - GRID_COLS; i++){
        data_input_MGVF >> up_MGVF[i];
        data_input_I >> up_I[i];
    }
    for(size_t i = 0; i < GRID_COLS; i++){
        data_input_MGVF >> up_MGVF[i + test_size/2 - GRID_COLS];
        down_MGVF[i] = up_MGVF[i + test_size/2 - GRID_COLS];
        
        data_input_I >> up_I[i + test_size/2 - GRID_COLS];
    }
    for(size_t i = 0; i < GRID_COLS; i++){
        data_input_MGVF >> up_MGVF[i + test_size/2];
        down_MGVF[i + GRID_COLS] = up_MGVF[i + test_size/2];
        
        data_input_I >> down_I[i];
    }
    for(size_t i = 0; i < test_size/2 - GRID_COLS; i++){
        data_input_MGVF >> down_MGVF[i + 2*GRID_COLS];
        data_input_I >> down_I[i + GRID_COLS];
    }


    data_input_MGVF.seekg(std::ios::beg);
    for(size_t i = 0; i < test_size; i++){
        check_output >> check_results[i];
    }
    
    data_input_MGVF.close();
    data_input_I.close();
    check_output.close();

    return 0;
}
///////////////////VERIFY FUNCTION///////////////////////////////////
bool verify(float* up_results, float* down_results, std::vector<float> check_results) {
    bool match = true;
    std::ofstream out("output.data");
    out.precision(18);
    out << std::fixed;

    for(size_t i = 0; i < test_size/2; i++){
        out << up_results[i];
        out << "\n";
        if(up_results[i] != check_results[i]){
            std::cout << "Unmatch in up_results[" << i << "]: " << up_results[i] << " != " << check_results[i] << "!\n";
            match = false;
        }
    }
    out << "\%\%\n";
    bool match2 =true;
    for(size_t i = 0; i < test_size/2; i++){
        out << down_results[i];
        out << "\n";
        if(down_results[i] != check_results[i + test_size/2]){
            std::cout << "Unmatch in down_results[" << i << "]: " << down_results[i] << " != " << check_results[i + test_size/2] << "!\n";
            match2 = false;
        }
    }

    std::cout << "TEST " << (match&&match2 ? "PASSED" : "FAILED") << std::endl;
    return match;
}
////////MAIN FUNCTION//////////
int main(int argc, char** argv) {
    unsigned int size = test_size;
    unsigned int size_mgvf_input = test_size/2 + GRID_COLS;
    unsigned int half_size = test_size/2;

    // I/O Data Vectors
    std::vector<float, aligned_allocator<float> > up_MGVF(size_mgvf_input);
    std::vector<float, aligned_allocator<float> > up_I(half_size);
    std::vector<float, aligned_allocator<float> > down_MGVF(size_mgvf_input);
    std::vector<float, aligned_allocator<float> > down_I(half_size);
    std::vector<float, aligned_allocator<float> > up_results(half_size);
    std::vector<float, aligned_allocator<float> > down_results(half_size);
    std::vector<float> check_results(size);

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
    cl::Kernel up_kernel;
    cl::Kernel down_kernel;

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
            OCL_CHECK(err, up_kernel = cl::Kernel(program, "up_kernel", &err));
            OCL_CHECK(err, down_kernel = cl::Kernel(program, "down_kernel", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Two kernel MGVF Algorithm with Input size:" << GRID_ROWS << " * " << GRID_COLS << std::endl;

    // Feed input
    reset(up_MGVF.data(), up_I.data(), down_MGVF.data(), down_I.data(), up_results.data(), down_results.data(), check_results.data());

    // Running the kernel

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication

    cl_mem_ext_ptr_t ptr_up_MGVF, ptr_up_I, ptr_up_results, ptr_down_MGVF, ptr_down_I, ptr_down_results;

    // For Allocating Buffer to specific Global Memory PC, user has to use
    // cl_mem_ext_ptr_t
    // and provide the PCs
    ptr_up_results.obj = up_results.data();
    ptr_up_results.param = 0;
    ptr_up_results.flags = pc[0];

    ptr_up_MGVF.obj = up_MGVF.data();
    ptr_up_MGVF.param = 0;
    ptr_up_MGVF.flags = pc[0];

    ptr_up_I.obj = up_I.data();
    ptr_up_I.param = 0;
    ptr_up_I.flags = pc[0];

    ptr_down_results.obj = down_results.data();
    ptr_down_results.param = 0;
    ptr_down_results.flags = pc[0];

    ptr_down_MGVF.obj = down_MGVF.data();
    ptr_down_MGVF.param = 0;
    ptr_down_MGVF.flags = pc[0];

    ptr_down_I.obj = down_I.data();
    ptr_down_I.param = 0;
    ptr_down_I.flags = pc[0];

    OCL_CHECK(err, cl::Buffer buffer_up_MGVF(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, size_mgvf_input*sizeof(float), &ptr_up_MGVF,
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_up_I(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, half_size*sizeof(float), &ptr_up_I,
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_down_MGVF(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, size_mgvf_input*sizeof(float), &ptr_down_MGVF,
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_down_I(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, half_size*sizeof(float), &ptr_down_I,
                                         &err));                                     
                                         
    OCL_CHECK(err, cl::Buffer buffer_up_results(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, half_size*sizeof(float), &ptr_up_results,
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_down_results(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, half_size*sizeof(float), &ptr_down_results,
                                         &err));                                     

/*
    OCL_CHECK(err, cl::Buffer buffer_up_MGVF(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size_mgvf_input*sizeof(float), up_MGVF.data(),
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_up_I(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, half_size*sizeof(float), up_I.data(),
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_down_MGVF(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size_mgvf_input*sizeof(float), down_MGVF.data(),
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_down_I(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, half_size*sizeof(float), down_I.data(),
                                         &err));                                     
                                         
    OCL_CHECK(err, cl::Buffer buffer_up_results(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, half_size*sizeof(float), up_results.data(),
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_down_results(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, half_size*sizeof(float), down_results.data(),
                                         &err));                                     
*/

    // Setting Kernel Arguments
    OCL_CHECK(err, err = up_kernel.setArg(0, buffer_up_results));
    OCL_CHECK(err, err = up_kernel.setArg(1, buffer_up_MGVF));
    OCL_CHECK(err, err = up_kernel.setArg(2, buffer_up_I));

    OCL_CHECK(err, err = down_kernel.setArg(0, buffer_down_results));
    OCL_CHECK(err, err = down_kernel.setArg(1, buffer_down_MGVF));
    OCL_CHECK(err, err = down_kernel.setArg(2, buffer_down_I));

    // Copy input data to device global memory

    printf("Write buffer!\n");

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_up_MGVF, buffer_up_I, buffer_down_MGVF, buffer_down_I}, 0 /* 0 means from host*/));
    q.finish();

    printf("Write finished!\n");

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(up_kernel));
    OCL_CHECK(err, err = q.enqueueTask(down_kernel));
    q.finish();

    // Copy Result from Device Global Memory to Host Local Memory
    
    printf("Read buffer!\n");
    
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_up_MGVF, buffer_down_MGVF}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    // OpenCL Host Code Ends
    printf("Read finished!\n");

    // Compare the device results with software results
    bool match = verify(up_MGVF.data(), down_MGVF.data() + GRID_COLS, check_results);

    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}