// The actual baseline of this code was written in ChatGPT...
// Thanks ChatGPT!

use crate::kernel_decl::{KernelProcedure, Kernels};

use super::kernel_decl::{Expression, Value};

impl Expression {

    pub fn simplify (expr: Expression) -> Expression {
        use Expression::*;

        match expr {
            Val { .. } => expr,

            Add { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| x + y, "add", Expression::make_add),
            Minus { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| x - y, "minus", Expression::make_minus),
            Mult { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| x * y, "mult", Expression::make_mult),
            Div { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| x / y, "div", Expression::make_div),
            Remainder { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| x % y, "remainder", Expression::make_remainder),
            ShiftRight { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| x >> y, "shiftleft", Expression::make_shiftleft),
            ShiftLeft { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| x << y, "shiftright", Expression::make_shiftright),
            BitwiseAnd { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| x & y, "bitwiseand", Expression::make_bitwiseand),
            MoreThan { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| (x > y) as i32, "morethan", Expression::make_more_than),
            LessThan { a, b } => Expression::simplify_binary_op(*a, *b, |x, y| (x < y) as i32, "lessthan", Expression::make_less_than),
        }
    }

    fn simplify_binary_op<F>(
        a: Expression,
        b: Expression,
        op: F,
        op_str: &str,
        constructor: fn(Expression, Expression) -> Expression,
    ) -> Expression
    where
        F: Fn(i32, i32) -> i32,
    {
        let a_simplified = Expression::simplify(a);
        let b_simplified = Expression::simplify(b);

        // ================== Constant Evaluation ================== 
        match (&a_simplified, &b_simplified) {
            // const eval
            (
                Expression::Val { v: Value::Constant { val: va } },
                Expression::Val { v: Value::Constant { val: vb } },
            ) => {
                return Expression::make_const(op(*va, *vb))
            },
            _ => {}
        }

        // ================== Equation identities ================== 
        if op_str == "add" {
            // x + 0 = x 
            match (&a_simplified, &b_simplified) {
                (Expression::Val { v: Value::Constant { val: 0 } }, expr) => { return expr.clone() },
                (expr, Expression::Val { v: Value::Constant { val: 0 } }) => { return expr.clone() },
                _ => {}
            }
        }

        if op_str == "minus" {
            // x - 0 = x
            match (&a_simplified, &b_simplified) {
                (expr, Expression::Val { v: Value::Constant { val: 0 } }) => { return expr.clone() },
                _ => {}
            }
        }

        if op_str == "mult" {
            // x * 1 = x
            match (&a_simplified, &b_simplified) {
                (Expression::Val { v: Value::Constant { val: 1 } }, expr) => { return expr.clone() },
                (expr, Expression::Val { v: Value::Constant { val: 1 } }) => { return expr.clone() },
                _ => {}
            }

            // x * 2^b = x << b (we assume that all integers are positive, not necessary true for negative)
            match (&a_simplified, &b_simplified) {
                (Expression::Val { v: Value::Constant { val } }, expr) => {
                    let log_two_res = (*val as f32).log2();
                    if log_two_res.fract() == 0.0 {
                        return Expression::make_shiftleft(
                            expr.clone(),
                            Expression::make_const(log_two_res as i32),
                        );
                    }
                },
                (expr, Expression::Val { v: Value::Constant { val } }) => {
                    let log_two_res = (*val as f32).log2();
                    if log_two_res.fract() == 0.0 {
                        return Expression::make_shiftleft(
                            expr.clone(),
                            Expression::make_const(log_two_res as i32),
                        );
                    }
                },
                _ => {}
            }
        }

        if op_str == "bitwiseand" {
            match (&a_simplified, &b_simplified) {
                (_, Expression::Val { v: Value::Constant { val: 0 } }) => { return Expression::make_const(0) },
                _ => {}
            }
        }

        if op_str == "div" {
            // x / 1 = x
            match (&a_simplified, &b_simplified) {
                (expr, Expression::Val { v: Value::Constant { val: 1 } }) => { return expr.clone() },
                _ => {}
            }

            // x / (2^b) = x >> b (we assume all integers are positive; not necessary true for negative)
            match (&a_simplified, &b_simplified) {
                (Expression::Val { v: Value::Constant { val } }, expr) => {
                    let log_two_res = (*val as f32).log2();
                    if log_two_res.fract() == 0.0 {
                        return Expression::make_shiftright(
                            expr.clone(),
                            Expression::make_const(log_two_res as i32),
                        );
                    }
                },
                (expr, Expression::Val { v: Value::Constant { val } }) => {
                    let log_two_res = (*val as f32).log2();
                    if log_two_res.fract() == 0.0 {
                        return Expression::make_shiftright(
                            expr.clone(),
                            Expression::make_const(log_two_res as i32),
                        );
                    }
                },
                _ => {}
            }
            
        }

        
        /*
        fn make_opt_remainder (a: Expression, b: Expression) -> Expression {
            if let Some(b_val) = b.get_const() {
                let log_two_res = (b_val as f32).log2();
                if log_two_res.fract() == 0.0 {
                    return Expression::make_bitwiseand(
                        a, 
                        Expression::make_const((1 << (log_two_res as i32)) - 1)
                    )
                }
            }
            Expression::make_remainder(a, b)
        }
        */

        // ======== if b >= a at x % a % b, it's equivalent to x % a +==========
        if op_str == "remainder" {
            match (&a_simplified, &b_simplified) {
                (
                    Expression::Remainder { a, b },
                    Expression::Val { v: Value::Constant { val: val_two } }
                ) => {
                    match &**b {
                        Expression::Val { v: Value::Constant { val: val_one } } => {
                            if val_one == val_two {
                                return Expression::make_remainder(*a.clone(), Expression::Val { v: Value::Constant { val: *val_one } })
                            }
                        },
                        _ => {}
                    }
                },
                _ => {}
            }
        }

        // if not, just return the orig expression
        constructor(a_simplified, b_simplified)
    }
}



// M (id: ax, access: #global)  =  CS (V: 0.1)  Multiply (128)  M (id: ax, access: (#global % 128))
// there are some cases like this where (#global % 128) can be simplified to #global because of the size of the elw operation
// this only applies to unary, binary operations, and movement (dot prod, reduce, etc. invalid)
pub fn could_replace (expr: &Expression, s: usize) -> Option<&Box<Expression>> {
    match expr {
        Expression::Remainder { a, b } => {
            match &**b {
                Expression::Val { v: Value::Constant { val } } => {
                    if (*val as usize) == s {
                        return Some(a)
                    }
                },
                _ => {}
            }
        },
        _ => {}
    }

    None
}

pub fn simplify_global_expr (kernel_proc: &mut KernelProcedure) {
    kernel_proc.step_cmd(&mut |proc, idx| {
        let cmd = proc.get_mut(*idx).unwrap(); 
        if let Some(p) = cmd.fus_get_mut_kernels() {
            for kernel in p.iter_mut() {
                
                let mut s = 0;
                if let Kernels::Unary { size, .. } = kernel { s = *size; }
                if let Kernels::Binary { size, .. } = kernel { s = *size; }
                if let Kernels::Movement { size, .. } = kernel { s = *size; }

                if s == 0 { continue; }

                for expr in kernel.get_any_mut_access_expr() {
                    if let Some(a) = could_replace(expr, s) {
                        *expr = *a.clone();
                    }
                }
            }
        } else {
            let mut s = 0;
            if let Kernels::Unary { size, .. } = cmd { s = *size; }
            if let Kernels::Binary { size, .. } = cmd { s = *size; }
            if let Kernels::Movement { size, .. } = cmd { s = *size; }

            if s == 0 { return true; }

            for expr in cmd.get_any_mut_access_expr() {
                if let Some(a) = could_replace(expr, s) {
                    *expr = *a.clone();
                }
            }
        }

        true
    });
}