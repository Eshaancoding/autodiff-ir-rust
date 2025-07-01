use crate::{devices::{context::OpenCLContext, fuse_elw::cl_elw_kernels_to_body, helper::get_inputs_args}, kernel_decl::{Input, Kernels, Output}};

pub fn execute_fuse_dp_elw (opencl_context: &mut OpenCLContext, cmd: &Kernels) {
    match cmd {
        Kernels::DPElwExpr { id, kernels, a_shape, res_shape, .. } => {
            // get important information
            let batch_size = a_shape.0;            
            let input_size = a_shape.1; 
            let output_size = res_shape.1;

            let kernel_name = format!("_{}", id);
            let parsed_args = get_inputs_args(cmd.get_inputs(), cmd.get_outputs());

            let (
                buffers, 
                mut e_kernel, 
                queue
            ) = opencl_context.get_kernel(&kernel_name, || {
                // get a, b, res from dot product
                let mut kernels = kernels.clone(); // not the best implementation... it's only called once anyways
                let a_dot: Input;
                let b_dot: Input;
                let res_dot: Output;

                if let Kernels::DotProd { a, b, res, .. } = kernels.remove(0) {
                    a_dot = a;
                    b_dot = b;
                    res_dot = res;
                } 
                else {
                    panic!("First command is not a DP operation!")
                }               

                let mut args = parsed_args.iter().map(|v| format!("__global float* {}", v)).collect::<Vec<String>>();
                args.push("int wA".to_string());
                args.push("int wB".to_string());

                format!(r#"
                __kernel void {} (
                    {} 
                )
                {{
                    int tx = get_global_id(0); // Column index in C; --> output size
                    int ty = get_global_id(1); // Row index in C;    --> batch size

                    float value = 0.0f;
                    for (int k = 0; k < wA; ++k) {{
                        int _x = ty;
                        int _y = k;
                        float elementA = {};

                        _x = k;
                        _y = tx;
                        float elementB = {};
                        value += elementA * elementB;
                    }}

                    int _x = ty;
                    int _y = tx;
                    float _temp_var = 0.0;
                    
                    {} = value;

                    int _global_id = ty * wB + tx;
                    {}
                }}
                "#, 
                    kernel_name, 
                    args.join(","),
                    a_dot.to_opencl(),
                    b_dot.to_opencl(),
                    res_dot.to_opencl(),
                    cl_elw_kernels_to_body(&kernels)
                )          
            });

            let kernel_event = unsafe {
                for id in parsed_args {
                    e_kernel.set_arg(buffers.get(&id).unwrap());
                }
                e_kernel.set_arg(&input_size);
                e_kernel.set_arg(&output_size);

                e_kernel
                    .set_global_work_size(output_size)
                    .set_global_work_size(batch_size)
                    .enqueue_nd_range(&queue)
                    .expect("Can't create execute kernel")
            };

            kernel_event.wait().expect("Can't wait for kernel event");
        },
        _ => {}
    }
}