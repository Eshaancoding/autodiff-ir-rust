// The actual baseline of this code was written in ChatGPT...
// Thanks ChatGPT!

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
        }

        if op_str == "div" {
            // x / 1 = x
            match (&a_simplified, &b_simplified) {
                (expr, Expression::Val { v: Value::Constant { val: 1 } }) => { return expr.clone() },
                _ => {}
            }
        }

        if op_str == "remainder" {
            // if b >= a at x % a % b, it's equivalent to x % a
            match (&a_simplified, &b_simplified) {
                (
                    Expression::Remainder { a, b},
                    Expression::Val { v: Value::Constant { val } },
                ) => {
                    let val_b = val.clone();

                    if let Some(val_a) = b.get_const() {
                        if val_b >= val_a {
                            return Expression::make_remainder(
                                *a.clone(), 
                                Expression::make_const(val_a)
                            )
                        }
                    }
                },
                _ => {}
            }
        }

        match (&a_simplified, &b_simplified) {
            
            // const eval
            (
                Expression::Val { v: Value::Constant { val: va } },
                Expression::Val { v: Value::Constant { val: vb } },
            ) => Expression::Val {
                v: Value::Constant { val: op(*va, *vb) },
            },
            
            // else, leave untouched
            _ => constructor(a_simplified, b_simplified),
        }
    }
}
