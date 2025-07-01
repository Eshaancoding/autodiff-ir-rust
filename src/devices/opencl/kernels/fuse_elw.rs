use crate::{devices::{binary::cl_binary_to_body, context::OpenCLContext, helper::get_inputs_args, movement::cl_movement_to_body, unary::cl_unary_to_body}, kernel_decl::Kernels};

pub fn cl_elw_kernels_to_body (kernels: &Vec<Kernels>) -> String {
    let mut body: String = String::new();
    for k in kernels {
        match k {
            Kernels::Binary { a, b, res, op, .. } => {
                body += cl_binary_to_body(a, b, res, op).as_str();
            },
            Kernels::Movement { a, res, .. } => {
                body += cl_movement_to_body(a, res).as_str();
            },
            Kernels::Unary { a, res, op, .. } => {
                body += cl_unary_to_body(a, res, op).as_str();
            }
            _ => panic!("Invalid kernel in elw fusion!")
        }
    }

    body
}

pub fn execute_elw_expr (opencl_context: &mut OpenCLContext, cmd: &Kernels) {
    match cmd {
        Kernels::ElwExpr { id, kernels, size } => {
            let kernel_name = format!("_{}", id);
            let inps = cmd.get_inputs();
            let outs = cmd.get_outputs();
            let parsed_args = get_inputs_args(inps, outs);

            let (
                buffers,
                mut e_kernel,
                queue
            )  = opencl_context.get_kernel(&kernel_name, || {
                format!(r#"
                    __kernel void {} (
                        {}
                    ) {{
                        const size_t _global_id = get_global_id(0);
                        float _temp_var = 0.0; 
                        {}
                    }}
                "#,
                    kernel_name,
                    parsed_args.iter().map(|v| format!("__global float* {}", v)).collect::<Vec<String>>().join(","),
                    cl_elw_kernels_to_body(kernels)
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