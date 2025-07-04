use crate::{devices::{context::OpenCLContext, fuse_elw::cl_elw_kernels_to_body, helper::get_inputs_args}, kernel_decl::{Input, Kernels, Output, ReduceOp}};

pub fn execute_fuse_reduce_elw (opencl_context: &mut OpenCLContext, cmd: &Kernels) {
    match cmd {
        Kernels::ReduceElwExpr { id, kernels, vec_size, reduce_size } => {
            let kernel_name = format!("_{}", id);
            let parsed_args = get_inputs_args(cmd.get_inputs(), cmd.get_outputs());

            let (
                buffers, 
                mut e_kernel, 
                queue
            ) = opencl_context.get_kernel(&kernel_name, || {
                // get a, op, and res fro mthe very first command
                let mut kernels = kernels.clone();
                let a_rd: Input;
                let res_rd: Output;
                let op_rd: ReduceOp;
                
                if let Kernels::Reduce { a, res, op, .. } = kernels.remove(0) {
                    a_rd = a;
                    res_rd = res;
                    op_rd = op;
                }
                else {
                    panic!("First command is not a Reduce operation!")
                }

                let mut args: Vec<String> = parsed_args.iter().map(|v| format!("__global float* {}", v)).collect();
                args.push("__local float* scratch".to_string());

                format!(r#"
                __kernel void {}(
                    {} 
                ) {{
                    
                    // global work size = local work size * number of groups
                    int _x = get_group_id(0);
                    int _y = get_local_id(0);
                    int local_size = get_local_size(0);
                    int total_size = get_global_size(0);

                    // Load data into local memory (not doing if clause check... okay?)
                    scratch[_y] = {};

                    barrier(CLK_LOCAL_MEM_FENCE); // waits until transfer to local memory is all finished

                    // Reduction in local memory
                    for (int offset = local_size / 2; offset > 0; offset >>= 1) {{
                        if (_y < offset) {{
                            {}
                        }}
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }}

                    // Write result of this work-group to partial_sums
                    if (_y == 0) {{
                        float _temp_var = 0.0;
                        {} = scratch[0];

                        int _global_id = _x; 
                        {}
                    }}
                }}
                "#, 
                    kernel_name, 
                    args.join(","),
                    a_rd.to_opencl(),
                    op_rd.to_opencl("scratch[_y]".to_string(), "scratch[_y + offset]".to_string()),
                    res_rd.to_opencl(),
                    cl_elw_kernels_to_body(&kernels)
                )          
            });

            let kernel_event = unsafe {
                for id in parsed_args {
                    e_kernel.set_arg(buffers.get(&id).unwrap());
                }
                e_kernel.set_arg_local_buffer(*reduce_size);

                e_kernel
                    .set_global_work_size(*reduce_size * *vec_size)
                    .set_local_work_size(*reduce_size)
                    .enqueue_nd_range(&queue)
                    .expect("Can't create execute kernel")
            };

            kernel_event.wait().expect("Can't wait for kernel event");
        },
        _ => {}
    }
}