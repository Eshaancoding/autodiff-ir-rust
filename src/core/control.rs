use crate::{Tensor, IRCmds};
use std::ops::Range;

use super::{autodiff, ir_b_add, ir_b_create_block, ir_b_id,  ir_b_set_main_block};


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
    let e_res = expr();
    let v = e_res.forward();
    if e_res.dim() != vec![1] {
        panic!("If expression must have dim of [1]")
    }
    
    let id = ir_b_id();
    let if_block_id = id.clone() + "_if";
    let end_block_id = id + "_if_end";
    
    ir_b_add(IRCmds::BRE {
        block_id: if_block_id.clone(),
        a: v.id.clone()
    }); // go to if block if true
    
    ir_b_add(IRCmds::BRZ {
        block_id: end_block_id.clone(),
        a: v.id
    }); // go to else block if not true
    
    // create if block
    ir_b_create_block(if_block_id);
    
    func();
    
    ir_b_add(IRCmds::BR {
        block_id: end_block_id.clone()
    }); // go to the end block after if statement finished
    
    // create end block    
    ir_b_create_block(end_block_id.clone());
    ir_b_set_main_block(end_block_id);
    
    // rest of the code will then execute
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
    // ============= EXPR ============= 
    let e_res = expr();
    let v = e_res.forward();
    if e_res.dim() != vec![1] {
        panic!("If expression must have dim of [1]")
    }
    
    let id = ir_b_id();
    let if_block_id = id.clone() + "_if";
    let else_block_id = id.clone() + "_else";
    let end_block_id = id + "_if_end";
    
    ir_b_add(IRCmds::BRE {
        block_id: if_block_id.clone(),
        a: v.id.clone()
    }); // go to if block if true
    
    ir_b_add(IRCmds::BRZ {
        block_id: else_block_id.clone(),
        a: v.id
    }); // go to else block if not true
    
    // ============= if block ============= 
    ir_b_create_block(if_block_id);
    
    func();
    
    ir_b_add(IRCmds::BR {
        block_id: end_block_id.clone()
    }); // go to the end block after if statement finished
    
    // ============= else block ============= 
    ir_b_create_block(else_block_id);
    
    func_else();
    
    ir_b_add(IRCmds::BR {
        block_id: end_block_id.clone()
    }); // go to the end block after if statement finished
    
    // ============= end block ============= 
    ir_b_create_block(end_block_id.clone());
    ir_b_set_main_block(end_block_id);
    
    // rest of the code will then execute
    
}
/**
 * Note that `func` must handle IR addition by `.forward` and `.backward`
 * `expr` automatically calls `.forward`; no IR handling at `expr`
 */
pub fn ir_while<E, I> (expr: E, func: I) 
where 
    E: FnOnce() -> Tensor,
    I: FnOnce() 
{
    // start while 
    let id = ir_b_id();
    let while_begin_id = id.clone() + "_while";
    let end_block_id = id + "_while_end";
    ir_b_add(IRCmds::BR { block_id: while_begin_id.clone()});

    ir_b_create_block(while_begin_id.clone());

    // test condition
    let e_res = expr();
    let v = e_res.forward();
    if e_res.dim() != vec![1] {
        panic!("while expression must have dim of [1]")
    }

    ir_b_add(IRCmds::BRZ {
        block_id: end_block_id.clone(),
        a: v.id
    }); // go to the end block if satisfies condition

    func(); // if not, just continue on with the rest of the function

    ir_b_add(IRCmds::BR { block_id: while_begin_id }); // go back to the start
    
    // ending block
    ir_b_create_block(end_block_id.clone());
    ir_b_set_main_block(end_block_id);
}

/**
 * Note that `func` must handle IR addition by `.forward` and `.backward`
 */
pub fn ir_for<F> (r: Range<i32>, func: F) 
where
    F: FnOnce(Tensor)
{
    // internally, it's just a while loop
    let x = autodiff::scalar(r.start as f64);
    let x_end = autodiff::const_val(r.end as f64, vec![1]);

    ir_while(|| x.less_than(&x_end), || {
        func(x.clone());

        let mut v = x.clone(); 
        v += 1.0;
        v.forward(); // will still apply to x's variable at IR; we just need to satisfy the borrow checker
    });
}