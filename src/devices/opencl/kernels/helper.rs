use crate::{kernel_decl::{Input, Output}};

pub fn get_inputs_args (inputs: Vec<&Input>, output: &Output) -> Vec<String> {
    let mut t: Vec<String> = vec![];
    for arg in inputs.iter() {
        match arg {
            Input::ConcatMatrix { .. } => {
                todo!()
            },
            Input::Constant { .. } => { },
            Input::Temp { } => {},
            Input::Mat { mat } => {
                t.push(mat.id.clone())
            }
        }
    }

    match output {
        Output::Mat { mat } => {
            let arg_name: String = format!("__global float* {}", mat.id);
            t.push(mat.id.clone());
        },
        Output::Temp {} => {}
    }

    t
}