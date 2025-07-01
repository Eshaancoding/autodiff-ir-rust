use crate::{devices::{context::OpenCLContext, helper::get_inputs_args}, kernel_decl::{Kernels, Output, Input}};

pub fn cl_movement_to_body (a: &Input, res: &Output) -> String {
    format!("{} = {};", res.to_opencl(), a.to_opencl())
}

pub fn execute_movement (opencl_context: &mut OpenCLContext, cmd: &Kernels) {
    match cmd {
        Kernels::Movement { id, a, res, size } => {
            let kernel_name = format!("_{}", id);
            let parsed_args = get_inputs_args(vec![a], res);

            let (
                buffers, 
                mut e_kernel, 
                queue
            ) = opencl_context.get_kernel(&kernel_name, || {
                format!(r#"
                __kernel void {} (
                    {}
                ) {{
                    const size_t _global_id = get_global_id(0);
                    {}
                }}
                "#, 
                    kernel_name, 
                    parsed_args.iter().map(|v| format!("__global float* {}", v)).collect::<Vec<String>>().join(","),
                    cl_movement_to_body(a, res)
                )          
            });

            let kernel_event = unsafe {
                for id in parsed_args {
                    e_kernel.set_arg(buffers.get(&id).unwrap());
                }
                e_kernel
                    .set_global_work_size(*size)
                    .enqueue_nd_range(&queue)
                    .expect("Can't create execute kernel")
            };

            kernel_event.wait().expect("Can't wait for kernel event");
        },
        _ => {}
    }
}