// Copyright (c) 2021 Via Technology Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_float, cl_int};
use std::ptr;

const PROGRAM_SOURCE: &str = r#"
__kernel void matrixMul(
    __global float* C,
    __global float* A,
    __global float* B,
    int wA, // Width of A (and height of B) --> Input size
    int wB  // Width of B                   --> output size
)
{
    int tx = get_global_id(0); // Column index in C; --> output size
    int ty = get_global_id(1); // Row index in C;    --> batch size

    float value = 0.0f;
    for (int k = 0; k < wA; ++k) {
        int _x = ty;
        int _y = k;
        float elementA = A[(_x << 3) + _y];

        _x = k;
        _y = tx;
        float elementB = B[(_x * 6) + _y];
        value += elementA * elementB;
    }

    int _x = ty;
    int _y = tx;
    C[_x * 6 + _y] = value;

}"#;

const KERNEL_NAME: &str = "matrixMul";

pub fn opencl_matmul () {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU).expect("Can't get device")
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    /////////////////////////////////////////////////////////////////////
    // Fill data data

    const BATCH_SIZE: usize = 4;
    const INPUT_SIZE: usize = 8;    
    const OUTPUT_SIZE: usize = 6;

    const A_SIZE: usize = BATCH_SIZE * INPUT_SIZE;
    const W_SIZE: usize = INPUT_SIZE * OUTPUT_SIZE;
    const RES_SIZE: usize = BATCH_SIZE * OUTPUT_SIZE; 

    // fill a data
    let mut a_data: [cl_float; A_SIZE] = [0.0; A_SIZE];
    for b in 0..BATCH_SIZE {
        for i in 0..INPUT_SIZE {
            a_data[b*INPUT_SIZE+i] = (b*INPUT_SIZE+i) as cl_float;
        }
    } 

    // fill w data
    let mut w_data: [cl_float; W_SIZE] = [0.0; W_SIZE];
    for i in 0..INPUT_SIZE {
        for o in 0..OUTPUT_SIZE {
            w_data[i*OUTPUT_SIZE+o] = 1.0 as cl_float;
        }
    }

    println!("a_data: {:?}", a_data);
    println!("w_data: {:?}", w_data);

    /////////////////////////////////////////////////////////////////////
    // Create buffers and fill buffers
    let mut a_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, A_SIZE, ptr::null_mut()).expect("Create buffer error")
    };
    
    // Weight buffer 
    let mut w_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, W_SIZE, ptr::null_mut()).expect("Create buffer error")
    };

    // output buffer
    let res_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, RES_SIZE, ptr::null_mut()).expect("Create buffer error")
    };

    // ========== Write to the input memory ========== 
    let a_buffer_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut a_buffer, CL_BLOCKING, 0, &a_data, &[]).expect("Write buffer error") 
    };
    a_buffer_write_event.wait().expect("Can't wait for A write");

    let w_buffer_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut w_buffer, CL_BLOCKING, 0, &w_data, &[]).expect("Write buffer error") 
    };
    w_buffer_write_event.wait().expect("Can't wait for W write");

    // Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let kernel_event = unsafe {

        // ********* LOCAL WORK SIZE = SIZE OF REDUCE/Y ********* 
        // ******** GLOBAL WORK SIZE = SIZE OF REDUCE/Y * VEC/X ********** 
        // then, the group ids should be allocated like that

        ExecuteKernel::new(&kernel)
            .set_arg(&res_buffer)
            .set_arg(&a_buffer)
            .set_arg(&w_buffer)
            .set_arg(&(INPUT_SIZE as cl_int))
            .set_arg(&(OUTPUT_SIZE as cl_int))
            .set_global_work_size(OUTPUT_SIZE)
            .set_global_work_size(BATCH_SIZE)
            .enqueue_nd_range(&queue).expect("Execute kernel error")
    };

    kernel_event.wait().expect("can't wait");

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.
    let mut results: [cl_float; RES_SIZE] = [-1.0; RES_SIZE];
    let read_event =
        unsafe { 
            queue.enqueue_read_buffer(&res_buffer, CL_BLOCKING, 0, &mut results, &[]).expect("Read buffer error")
        };

    // Wait for the read_event to complete.
    read_event.wait().expect("Wait expect");

    // find the results
    println!("results: {:#?}", results);
}