// reduction_host.c

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (1 << 20) // Number of elements (e.g., 1 million)
#define WG_SIZE 256 // Work-group size

/*

__kernel void reduce_sum(__global const float* input,
                         __local float* scratch,
                         const int length,
                         __global float* partial_sums) {
    int global_id = get_global_id(0);
    int local_id  = get_local_id(0);
    int group_id  = get_group_id(0);
    int local_size = get_local_size(0);

    // Load data into local memory
    float val = (global_id < length) ? input[global_id] : 0.0f;
    scratch[local_id] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in local memory
    for (int offset = local_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            scratch[local_id] += scratch[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result of this work-group to partial_sums
    if (local_id == 0) {
        partial_sums[group_id] = scratch[0];
    }
}
*/
// Helper to check OpenCL errors
#define CHECK_ERR(err, msg) \
    if (err != CL_SUCCESS) { fprintf(stderr, "%s: %d\n", msg, err); exit(1); }

int main() {
    // 1. Prepare input data
    float *h_input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f; // Example: all ones

    // 2. Get OpenCL platform/device/context/queue
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL); CHECK_ERR(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); CHECK_ERR(err, "clGetDeviceIDs");
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK_ERR(err, "clCreateContext");
    queue = clCreateCommandQueue(context, device, 0, &err); CHECK_ERR(err, "clCreateCommandQueue");

    // 3. Load and build kernel
    FILE *f = fopen("reduction_kernel.cl", "rb");
    fseek(f, 0, SEEK_END); size_t src_size = ftell(f); rewind(f);
    char *src = (char*)malloc(src_size + 1);
    fread(src, 1, src_size, f); src[src_size] = 0; fclose(f);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &err); CHECK_ERR(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build log on error
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "%s\n", log);
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "reduce_sum", &err); CHECK_ERR(err, "clCreateKernel");

    // 4. Create buffers
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), h_input, &err); CHECK_ERR(err, "clCreateBuffer input");
    int num_groups = (N + WG_SIZE - 1) / WG_SIZE;
    cl_mem d_partial = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(float), NULL, &err); CHECK_ERR(err, "clCreateBuffer partial");

    // 5. Set kernel arguments and launch first stage

    // input buffer
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input); CHECK_ERR(err, "clSetKernelArg 0");
    
    // local buffer (set to NULL??)
    err = clSetKernelArg(kernel, 1, WG_SIZE * sizeof(float), NULL); CHECK_ERR(err, "clSetKernelArg 1");
    
    // Length input (int)
    err = clSetKernelArg(kernel, 2, sizeof(int), &N); CHECK_ERR(err, "clSetKernelArg 2");
    
    // output buffer
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_partial); CHECK_ERR(err, "clSetKernelArg 3");

    size_t global = num_groups * WG_SIZE;
    size_t local = WG_SIZE;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL); CHECK_ERR(err, "clEnqueueNDRangeKernel");

    // 6. Read back partial sums
    float *h_partial = (float*)malloc(num_groups * sizeof(float));
    err = clEnqueueReadBuffer(queue, d_partial, CL_TRUE, 0, num_groups * sizeof(float), h_partial, 0, NULL, NULL); CHECK_ERR(err, "clEnqueueReadBuffer");

    // 7. Final reduction on host
    float sum = 0.0f;
    for (int i = 0; i < num_groups; ++i) sum += h_partial[i];

    printf("Sum = %f (expected %f)\n", sum, (float)N);

    // 8. Cleanup
    free(h_input); free(h_partial); free(src);
    clReleaseMemObject(d_input); clReleaseMemObject(d_partial);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return 0;
}
