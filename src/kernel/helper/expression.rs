use super::kernel_decl::{Expression, Value};


impl Expression {
    pub fn get_const (&self) -> Option<i32> {
        match self {
            Expression::Val { v } => {
                match v {
                    Value::Constant { val } => { Some(val.clone()) },
                    _ => { None }
                }
            },
            _ => { None }
        }
    }

    pub fn make_const (v: i32) -> Expression {
        Expression::Val { v: Value::Constant { val: v } }
    }

    pub fn make_add (a: Expression, b: Expression) -> Expression {
        Expression::Add { 
            a: Box::new(a),
            b: Box::new(b)
        }
    }
    
    pub fn make_minus (a: Expression, b: Expression) -> Expression {
        Expression::Minus { 
            a: Box::new(a),
            b: Box::new(b) 
        }
    }

    pub fn make_mult (a: Expression, b: Expression) -> Expression {
        Expression::Mult { 
            a: Box::new(a),
            b: Box::new(b) 
        }
    }

    pub fn make_div (a: Expression, b: Expression) -> Expression {
        Expression::Div { 
            a: Box::new(a), 
            b: Box::new(b) 
        } 
    }

    pub fn make_remainder (a: Expression, b: Expression) -> Expression {
        Expression::Remainder { 
            a: Box::new(a), 
            b: Box::new(b) 
        }
    }

    pub fn make_bitwiseand (a: Expression, b: Expression) -> Expression {
        Expression::BitwiseAnd { 
            a: Box::new(a),
            b: Box::new(b) 
        }
    }

    pub fn make_shiftright (a: Expression, b: Expression) -> Expression {
        Expression::ShiftRight { 
            a: Box::new(a), 
            b: Box::new(b) 
        }
    }

    pub fn make_shiftleft (a: Expression, b: Expression) -> Expression {
        Expression::ShiftLeft { 
            a: Box::new(a), 
            b: Box::new(b) 
        }
    }

    pub fn make_global () -> Expression {
        Expression::Val { v: Value::Global }
    }

    
}
