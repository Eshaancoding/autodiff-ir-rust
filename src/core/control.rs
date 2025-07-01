use crate::{core::add_to_dep, ir_b_create_temp_proc, ir_b_return_temp_proc, IRCmds, Tensor};
use std::ops::Range;

use super::{autodiff, ir_b_add};


/**
 * Note that `func` must handle IR addition by `.forward` and `.backward`
 * Aka all operations inside closure must also include `.forward` or `.backward`
 * Closure `expr` automatically calls `.forward`; no IR handling at `expr`
 */
pub fn ir_if<E, F> (expr: E, func: F) 
where 
    E: FnOnce() -> Tensor,
    F: FnOnce()
{
    // Also, make sure to add to dep list
    
    // add if evaluations to main 
    let v = expr().forward();
    add_to_dep(v.id.clone());    
    if v.dim != vec![1] { panic!("If expression must have dim of [1]") }
    
    // rest of the code will then execute
    ir_b_create_temp_proc();
    func();    
    let if_block = ir_b_return_temp_proc();

    // add if to main
    ir_b_add(IRCmds::If { 
        conditions: vec![(v.id, if_block)],
        else_proc: None
    });
}

/**
 * Note that `func` and `func_else` must handle IR addition by `.forward` and `.backward`
 * Aka all operations inside closure must also include `.forward` or `.backward`
 * Closure `expr` automatically calls `.forward`; no IR handling at `expr`
 */
pub fn ir_if_else<E, F, FE> (expr: E, func: F, func_else: FE) 
where 
    E: FnOnce() -> Tensor,
    F: FnOnce(),
    FE: FnOnce()
{
    // add if evaluations to main 
    let v = expr().forward();
    add_to_dep(v.id.clone());     // add to dependency list
    if v.dim != vec![1] { panic!("Expr must return a boolean") }
    
    // func
    ir_b_create_temp_proc();
    func();    
    let if_block = ir_b_return_temp_proc();

    ir_b_create_temp_proc();
    func_else();    
    let if_else_block = ir_b_return_temp_proc();

    // add if to main
    ir_b_add(IRCmds::If { 
        conditions: vec![(v.id, if_block)],
        else_proc: Some(if_else_block)
    });
}


/**
 * Note that `func` must handle IR addition by `.forward` and `.backward`
 * `expr` automatically calls `.forward`; no IR handling at `expr`
 * Note that in the body, you are responsible for updating var
 * Follows: while (var != 0) { func() }
 */
pub fn ir_while<I> (var: &Tensor, func: I) 
where 
    I: FnOnce() 
{
    // first, evaluate control expr
    let v = var.forward(); // first, evaluate control expr
    add_to_dep(v.id.clone());
    if v.dim != vec![1] { panic!("Expr must return a boolean.") } 
    
    // create func
    ir_b_create_temp_proc();
    func();
    let r = var.forward(); // re-evaluate var
    let main_func = ir_b_return_temp_proc();

    // add while loop
    ir_b_add(IRCmds::While { conditional_var: r.id, block: main_func });
}

/**
 * Note that `func` must handle IR addition by `.forward` and `.backward`
 */
pub fn ir_for<F> (r: Range<i32>, func: F) 
where
    F: FnOnce(&Tensor)
{
    // internally, it's just a while loop
    let mut x = autodiff::scalar(r.start as f32);
    let x_end = autodiff::constant(r.end as f32, vec![1]);

    let result = x.less_than(&x_end);
    
    ir_while(&result, || {
        func(&x);
        x += 1.0; 
    });
}