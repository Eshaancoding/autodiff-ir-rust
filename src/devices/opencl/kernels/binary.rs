use crate::kernel_decl::{BinaryOp, Input, Output};

impl BinaryOp {
    pub fn to_opencl (&self) -> String {
        match self {
            BinaryOp::Add => "+".to_string(),
            BinaryOp::Multiply => "-".to_string()
        }
    }
}

pub fn to_opencl_binary_body (a: &Input, b: &Input, res: &Output, op: &BinaryOp) -> String {
    format!("{} = {} {} {}", res.to_opencl(), a.to_opencl(), op.to_opencl(), b.to_opencl()).to_string()
}