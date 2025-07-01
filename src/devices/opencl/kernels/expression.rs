use crate::kernel_decl::{Expression, Value, Matrix, Input, Output};

impl Value {
    pub fn to_opencl (&self) -> String {
        match self {
            Value::Constant { val } => val.to_string(),
            Value::Global => "_global_id".to_string(),
            Value::X => "_x".to_string(),
            Value::Y => "_y".to_string()
        }
    }
}

impl Expression {
    pub fn to_opencl (&self) -> String {
        match self {
            Expression::Val { v } => v.to_opencl(),
            Expression::Add { a, b } => { format!("({} + {})", a.to_opencl(), b.to_opencl()).to_string() },
            Expression::Minus { a, b } => { format!("({} - {})", a.to_opencl(), b.to_opencl()).to_string() },
            Expression::Mult { a, b } => { format!("({} * {})", a.to_opencl(), b.to_opencl()).to_string() },
            Expression::Div { a, b } => { format!("({} / {})", a.to_opencl(), b.to_opencl()).to_string() },
            Expression::Remainder { a, b } => { format!("({} % {})", a.to_opencl(), b.to_opencl()).to_string() },
            Expression::ShiftRight { a, b } => { format!("({} >> {})", a.to_opencl(), b.to_opencl()).to_string() },
            Expression::ShiftLeft { a, b } => { format!("({} << {})", a.to_opencl(), b.to_opencl()).to_string() },
            Expression::BitwiseAnd { a, b } => { format!("({} & {})", a.to_opencl(), b.to_opencl()).to_string() },
            Expression::MoreThan { a, b } => { format!("({} > {})", a.to_opencl(), b.to_opencl()).to_string() },
            Expression::LessThan { a, b } => { format!("({} < {})", a.to_opencl(), b.to_opencl()).to_string() }
        }
    }
}

impl Matrix {
    pub fn to_opencl (&self) -> String {
        format!("{}[{}]", self.id, self.access.to_opencl()).to_string()
    }
}

impl Input {
    pub fn to_opencl (&self) -> String {
        match self {
            Input::Constant { val } => val.to_string(),
            Input::Mat { mat } => mat.to_opencl(),
            Input::ConcatMatrix { .. } => todo!(), // not doing this one yet because might be multi-lined or something
            Input::Temp => "_temp".to_string()
        }
    }
}

impl Output {
    pub fn to_opencl (&self) -> String {
        match self {
            Output::Mat { mat } => mat.to_opencl(),
            Output::Temp => "_temp".to_string()
        }
    }
}