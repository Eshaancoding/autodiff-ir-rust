use crate::kernel_decl::{Input, Output, UnaryOp};

impl UnaryOp {
    pub fn to_opencl (&self, a: String) -> String {
        match self {
            UnaryOp::Exp2 => format!("exp2({})", a),
            UnaryOp::Log2 => format!("log2({})", a),
            UnaryOp::Sin => format!("sin({})", a),
            UnaryOp::Neg => format!("-{}", a),
            UnaryOp::Recip => format!("1.0/{}", a),
            UnaryOp::Sqrt => format!("sqrt(fabs({}))", a),
            UnaryOp::EqualZero => format!("({} == 0.0f) ? 1.0 : 0.0", a),
            UnaryOp::MoreZero => format!("({} > 0.0f) ? 1.0 : 0.0", a),
            UnaryOp::LessZero => format!("({} < 0.0f) ? 1.0 : 0.0", a)
        }
    }
}

pub fn to_opencl_unary_body (a: &Input, res: &Output, op: &UnaryOp) -> String {
    format!("{} = {}", res.to_opencl(), op.to_opencl(a.to_opencl()))
}