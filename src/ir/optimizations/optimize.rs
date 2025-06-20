// use crate::ir_print;
use crate::{core::env_flags::disable_ir_opt, IRB};

use super::{helper::get_score, opts::*};

pub fn ir_optimize () {
    // skip opt if we don't want it
    if disable_ir_opt() { return; }
    
    let mut guard = IRB.lock().unwrap();
    let irb = guard.as_mut().expect("Can't unpack IRBuilder");
    
    /*
    // optimizations
    repeat_opt(&mut irb.proc);
    dep_opt(&mut irb.proc);  
    br_opt(&mut irb.proc);  
    mem_opt(&mut irb.proc);      // should be around last; doesn't delete any cmds
    
    // memory opts past this point has to account for usage of variables more than once.
    opeq_opt(&mut irb.proc);     
    
    // Loop were we apply different prox optimizations and see what has the highest score out of all the optimizations
    // "Aggressive" optimization. Applies every optimization until we can't improve anymore
    let mut prev_max_val: Option<usize> = None;
    loop {
        let mut prox_rev_copy = irb.proc.clone();
        prox_rev_opt(&mut prox_rev_copy);
        let prox_rev_score = get_score(&prox_rev_copy);

        let mut prox_copy = irb.proc.clone();
        prox_opt(&mut prox_copy);     // changing order of this results in different programs, but similar memory opts
        let prox_score = get_score(&prox_copy);
        
        let mut chain_copy = irb.proc.clone();
        chain_opt(&mut chain_copy);    
        let chain_score = get_score(&chain_copy);

        let max_val = prox_rev_score.max(prox_score).max(chain_score);
        if prev_max_val.is_some_and(|f| f >= max_val) { break; } // if best max val opts are worse or equal to prev, stop the loop

        if max_val == chain_score { irb.proc = chain_copy; }
        else if max_val == prox_score { irb.proc = prox_copy; }
        else if max_val == prox_rev_score { irb.proc = prox_rev_copy; }
        
        prev_max_val = Some(max_val);
    }

    dep_opt(&mut irb.proc);  // we may have deleted a few things, so it doesn't hurt to delete things that are not referenced again
    */

    drop(guard);
}