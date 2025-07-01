use crate::{devices::{context::OpenCLContext, helper::get_inputs_args}, kernel_decl::{BinaryOp, Input, Kernels, Output}};

impl BinaryOp {
    pub fn to_opencl (&self) -> String {
        match self {
            BinaryOp::Add => "+".to_string(),
            BinaryOp::Multiply => "*".to_string()
        }
    }
}

pub fn cl_binary_to_body (a: &Input, b: &Input, res: &Output, op: &BinaryOp) -> String {
    format!("{} = {} {} {};", res.to_opencl(), a.to_opencl(), op.to_opencl(), b.to_opencl()).to_string()
}


pub fn execute_binary (opencl_context: &mut OpenCLContext, cmd: &Kernels) {
    match cmd {
        Kernels::Binary { id, a, b, res, op, size } => {
            let kernel_name = format!("_{}", id);
            let parsed_args = get_inputs_args(vec![a, b], res);

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
                    cl_binary_to_body(a, b, res, op)
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