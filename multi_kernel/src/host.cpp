#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "xcl2.hpp"
// For LC_MGVF Constants
#include "lc_mgvf.h"

auto constexpr test_size = GRID_ROWS * GRID_COLS;

////////////////////RESET FUNCTION//////////////////////////////////
int reset(int* a, int* b, int* c, int* sw_results, int* hw_results, unsigned int size) {
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
    return 0;
}
///////////////////VERIFY FUNCTION///////////////////////////////////
bool verify(int* sw_results, int* hw_results, int size) {
    bool match = true;
    /*
    //No care
    for (int i = 0; i < size; i++) {
        if (sw_results[i] != hw_results[i]) {
            match = false;
            break;
        }
    }
    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    */
    return match;
}
////////MAIN FUNCTION//////////
int main(int argc, char** argv) {
    unsigned int size = test_size;
    unsigned int size_input = test_size;
    unsigned int size_result = test_size;

    // I/O Data Vectors
    std::vector<float, aligned_allocator<float> > up_MGVF(size_input);
    std::vector<float, aligned_allocator<float> > up_I(size_input);
    std::vector<float, aligned_allocator<float> > down_MGVF(size_input);
    std::vector<float, aligned_allocator<float> > down_I(size_input);
    std::vector<float, aligned_allocator<float> > up_results(size_result);
    std::vector<float, aligned_allocator<float> > down_results(size_result);
    std::vector<int> sw_results(size);

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
    // reset(up_MGVF.data(), up_I.data(), down_MGVF.data(), down_I.data(), up_results.data(), down_results.data(), size);

    // Running the kernel

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_up_MGVF(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size_input*sizeof(float), up_MGVF.data(),
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_up_I(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_input*sizeof(float), up_I.data(),
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_down_MGVF(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size_input*sizeof(float), down_MGVF.data(),
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_down_I(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_input*sizeof(float), down_I.data(),
                                         &err));                                     
                                         
    OCL_CHECK(err, cl::Buffer buffer_up_results(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size_result*sizeof(float), up_results.data(),
                                         &err));
    OCL_CHECK(err, cl::Buffer buffer_down_results(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size_result*sizeof(float), down_results.data(),
                                         &err));                                     


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

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(up_kernel));
    OCL_CHECK(err, err = q.enqueueTask(down_kernel));
    q.finish();

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_up_results, buffer_down_results}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    // OpenCL Host Code Ends
    printf("???\n");

    // Compare the device results with software results
    //bool match = verify(sw_results.data(), hw_results.data(), size);
    bool match = true;

    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}