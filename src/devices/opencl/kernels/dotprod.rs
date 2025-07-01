use crate::{devices::{context::OpenCLContext, helper::get_inputs_args}, kernel_decl::Kernels};

pub fn execute_dot_prod (opencl_context: &mut OpenCLContext, cmd: &Kernels) {
    match cmd {
        Kernels::DotProd { id, a, b, res, a_shape, res_shape, .. } => {
            let batch_size = a_shape.0;            
            let input_size = a_shape.1; 
            let output_size = res_shape.1;

            let kernel_name = format!("_{}", id);
            let parsed_args = get_inputs_args(vec![a, b], res);

            let (
                buffers, 
                mut e_kernel, 
                queue
            ) = opencl_context.get_kernel(&kernel_name, || {
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
                    {} = value;
                }}
                "#, 
                    kernel_name, 
                    args.join(","),
                    a.to_opencl(),
                    b.to_opencl(),
                    res.to_opencl()
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