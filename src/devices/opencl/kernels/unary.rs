use crate::{devices::{context::OpenCLContext, helper::get_inputs_args}, kernel_decl::{UnaryOp, Input, Kernels, Output}};

impl UnaryOp {
    pub fn to_opencl (&self, a: String) -> String {
        match self {
            UnaryOp::Exp2 => format!("exp2({})", a),
            UnaryOp::Log2 => format!("log2({})", a),
            UnaryOp::Sin => format!("sin({})", a),
            UnaryOp::Neg => format!("-{}", a),
            UnaryOp::Recip => format!("1.0/{}", a),
            UnaryOp::Sqrt => format!("sqrt(fabs({}))", a),
            UnaryOp::EqualZero => format!("(({} == 0.0f) ? 1.0 : 0.0)", a),
            UnaryOp::MoreZero => format!("(({} > 0.0f) ? 1.0 : 0.0)", a),
            UnaryOp::LessZero => format!("(({} < 0.0f) ? 1.0 : 0.0)", a)
        }
    }
}

pub fn cl_unary_to_body (a: &Input, res: &Output, op: &UnaryOp) -> String {
    format!("{} = {};", res.to_opencl(), op.to_opencl(a.to_opencl()))
}

pub fn execute_unary (opencl_context: &mut OpenCLContext, cmd: &Kernels) {
    match cmd {
        Kernels::Unary { id, a, res, op, size } => {
            let kernel_name = format!("_{}", id);
            let parsed_args = get_inputs_args(vec![a], vec![res]);

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
                    cl_unary_to_body(a, res, op)
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