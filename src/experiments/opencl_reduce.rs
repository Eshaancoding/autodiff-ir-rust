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
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_event, cl_float};
use std::ptr;
use std::sync::Arc;

const PROGRAM_SOURCE: &str = r#"
__kernel void reduce_sum(
    __global const float* input,
    __local float* scratch,
    __global float* partial_sums
) {
    
    // global work size = local work size * number of groups
    int _x = get_group_id(0);
    int _y = get_local_id(0);
    int local_size = get_local_size(0);
    int total_size = get_global_size(0);

    // Load data into local memory
    int idx = ((_x << 3) + _y);
    float val = (idx < total_size) ? input[idx] : 0.0f;
    scratch[_y] = val;

    barrier(CLK_LOCAL_MEM_FENCE); // waits until transfer to local memory is all finished

    // Reduction in local memory
    for (int offset = local_size / 2; offset > 0; offset >>= 1) {
        if (_y < offset) {
            scratch[_y] += scratch[_y + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result of this work-group to partial_sums
    if (_y == 0) {
        partial_sums[_x] = scratch[0];
    }
}"#;

const KERNEL_NAME: &str = "reduce_sum";

pub fn opencl_ruduce () {
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
    // Compute data

    // The input data
    const VEC_SIZE: usize = 4;
    const REDUCE_SIZE: usize = 8;
    
    const TOTAL_SIZE: usize = VEC_SIZE * REDUCE_SIZE;
    
    /*
    let mut input_data: [cl_float; TOTAL_SIZE] = [0.0; TOTAL_SIZE];
    
    for v in 0..VEC_SIZE {
        for r in 0..REDUCE_SIZE {
            input_data[v*REDUCE_SIZE+r] = v as cl_float
        }
    } 

    input_data[TOTAL_SIZE-1] = 4.0;
    */

    let input_data: Arc<Vec<f32>> = Arc::new(vec![1.0; TOTAL_SIZE]);
    // for v in 0..VEC_SIZE {
    //     for r in 0..REDUCE_SIZE {
    //         input_data[v*REDUCE_SIZE+r] = v as cl_float
    //     }
    // } 

    // input_data[TOTAL_SIZE-1] = 4.0;
    

    println!("input data: {:?}", input_data);

    // Create OpenCL device buffers
    let mut input_mem = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, TOTAL_SIZE, ptr::null_mut()).expect("Create buffer error")
    };
    
    // Output buffer 
    let output_mem = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, VEC_SIZE, ptr::null_mut()).expect("Create buffer error")
    };
    
    // ========== Write to the input memory ========== 
    let _x_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut input_mem, CL_BLOCKING, 0, &input_data, &[]).expect("Write buffer error") 
    };

    _x_write_event.wait().expect("Can't with for x writing");

    // Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let kernel_event = unsafe {

        // ********* LOCAL WORK SIZE = SIZE OF REDUCE/Y ********* 
        // ******** GLOBAL WORK SIZE = SIZE OF REDUCE/Y * VEC/X ********** 
        // then, the group ids should be allocated like that

        ExecuteKernel::new(&kernel)
            .set_arg(&input_mem)
            .set_arg_local_buffer(REDUCE_SIZE)
            .set_arg(&output_mem)
            .set_global_work_size(TOTAL_SIZE)
            .set_local_work_size(REDUCE_SIZE)
            // .set_wait_event(&_x_write_event)
            .enqueue_nd_range(&queue).expect("Execute kernel error")
    };

    kernel_event.wait().expect("can't wait");

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.
    // let mut results: [cl_float; VEC_SIZE] = [-1.0; VEC_SIZE];
    let mut results: Vec<cl_float> = vec![-1.0; VEC_SIZE];
    let read_event =
        unsafe { 
            queue.enqueue_read_buffer(&output_mem, CL_BLOCKING, 0, &mut results, &events).expect("Read buffer error")
        };

    // Wait for the read_event to complete.
    read_event.wait().expect("Wait expect");

    // find the results
    println!("results: {:#?}", results);

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start().expect("Start time error");
    let end_time = kernel_event.profiling_command_end().expect("End time error");
    let duration = end_time - start_time;
}