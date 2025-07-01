use crate::kernel_decl::{Input, Output, ReduceOp};

impl ReduceOp {
    pub fn to_opencl (&self, orig: String, new: String) -> String {
        match self {
            ReduceOp::Sum => format!("{} += {}", orig, new),
            ReduceOp::Max => format!("{} = max({}, {})", orig, orig, new),
        }
    }
}