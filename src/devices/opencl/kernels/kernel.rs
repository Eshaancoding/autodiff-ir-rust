use opencl3::{context::Context, kernel::Kernel as CLKernel, program::Program};
use crate::{devices::{binary::to_opencl_binary_body, unary::to_opencl_unary_body}, kernel_decl::{Input, Kernels}};

pub struct OpenCLKernelResult {
    pub kernel: CLKernel,
    pub work_size: usize,
    pub program_src: String
}

fn get_arg_list (inputs: Vec<&Input>) -> String {
    let mut l: Vec<String> = vec![];

    for &i in inputs.iter() {
        match i {
            Input::Mat { mat } => {
                let st = format!("__global float* {}", mat.id);
                if !l.contains(&st) { l.push(st); }
            },
            _ => {}
        }
    }

    l.join(",\n")
}

impl Kernels {
    pub fn to_opencl (&self, context: &Context) -> Option<OpenCLKernelResult> {
        let body = match self {
            Kernels::Binary { a, b, res, op, .. } => {
                Some(to_opencl_binary_body(a, b, res, op))
            },
            Kernels::Unary { a, res, op, .. } => {
                Some(to_opencl_unary_body(a, res, op))
            }
            _ => { None }
        };

        if let Some(src_body) = body {
            let name: String = format!("_{}", self.get_kernel_id().unwrap().to_string()).to_string();

            let program_src = format!(
                r#"
                kernel void {} ({}) {{
                    {}
                }}
                "#, 
                name, 
                get_arg_list(self.get_inputs()), 
                src_body
            );

            let program = Program::create_and_build_from_source(context, &program_src, "")
                .expect("Can't build program");

            let kernel = CLKernel::create(&program, &name)
                .expect("Can't create kernel");
                

            Some(OpenCLKernelResult { 
                kernel, 
                work_size: self.get_kernel_work_size().unwrap(),
                program_src: program_src.clone()
            })
        } 
        else {
            None
        }
    }
}