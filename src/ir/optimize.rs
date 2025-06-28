// use crate::ir_print;
use crate::{core::env_flags::disable_ir_opt,  IRB};
use crate::ir::opts::{
    const_begin, dep_opt, mem_opt, opeq_opt, prox_opt, prox_rev_opt, repeat_opt, track_var_changed
};


pub fn ir_optimize () {
    // Very basic IR optimizations

    // skip opt if we don't want it
    if disable_ir_opt() { return; }
    
    let mut guard = IRB.lock().unwrap();
    let irb = guard.as_mut().expect("Can't unpack IRBuilder");
    let var_changed = track_var_changed(&mut irb.proc);
    
    // Dept optimizations --> deletes unused variables
    loop {
        let deleted = dep_opt(&mut irb.proc, &var_changed);
        if deleted == 0 { break; }
    }   

    // Repeat optimizations --> deletes repetitive operations
    loop {
        let deleted = repeat_opt(&mut irb.proc, &var_changed);
        if deleted == 0 { break; }
    }
    
    // Set constants to the front --> no optimizations; just nicer to look at IR.
    const_begin(&mut irb.proc);

    // also do graph optimizations here for nicer simplification
    // constant evalutation
    // *= opts
    // probably more ideas in TODO
    // etc.





    // Memory optimization --> replace unnecessary ids; less memory allocated at kernel level
    
    // switch memory opts into kernel size
    /*
    mem_opt(&mut irb.proc, &var_changed);

    // Opeq optimization --> after memory optimization, there's often times where m = m * eq would exist. This could be simplified to m *= eq
    opeq_opt(&mut irb.proc); 
    
    // Re-ordering optimizations --> tries to group together repeated memory locations together for better caching + kernel fusion
    // Kernel fusion **relies** on the specific order of operations to determine fused operations
    let mut prev_max_val: Option<usize> = None;
    loop {
        // copy, change, score
        let mut prox_rev_copy = irb.proc.clone();
        prox_rev_opt(&mut prox_rev_copy);
        let prox_rev_score = get_score(&mut prox_rev_copy);

        // copy, change, score
        let mut prox_copy = irb.proc.clone();
        prox_opt(&mut prox_copy);                   // TODO: Doesn't have safegaurd for swaps in between IF/While. Still okay as it pasts the tests? 
        let prox_score = get_score(&mut prox_copy);

        // Calculate max
        let max_val = prox_rev_score.max(prox_score);
        if prev_max_val.is_some_and(|f| f >= max_val) { break; } 
        // if best max val opts are worse or equal to prev, stop the loop -> achieved maximum

        // set to current proc
        if max_val == prox_score { irb.proc = prox_copy; }
        else if max_val == prox_rev_score { irb.proc = prox_rev_copy; }
        
        prev_max_val = Some(max_val);
    }
    */

    drop(guard);
}