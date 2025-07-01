use crate::{kernel_decl::{Input, Output}};

// this could be replaced by kernel helper if I am being honest
pub fn get_inputs_args (inputs: Vec<&Input>, output: Vec<&Output>) -> Vec<String> {
    let mut t: Vec<String> = vec![];
    for arg in inputs.iter() {
        match arg {
            Input::ConcatMatrix { .. } => {
                todo!()
            },
            Input::Constant { .. } => { },
            Input::Temp { } => {},
            Input::Mat { mat } => {
                if !t.contains(&mat.id) {
                    t.push(mat.id.clone())
                }
            }
        }
    }

    for out in output.iter() {
        match out {
            Output::Mat { mat } => {
                if !t.contains(&mat.id) {
                    t.push(mat.id.clone());
                }
            },
            Output::Temp {} => {}
        }
    }

    t
}