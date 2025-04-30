// use crate::ir_print;
use crate::IRB;

use super::{helper::get_score, opts::*};

pub fn ir_optimize () {

    let mut guard = IRB.lock().unwrap();
    let irb = guard.as_mut().expect("Can't unpack IRBuilder");
    
    // optimizations
    repeat_opt(&mut irb.cmds);
    dep_opt(&mut irb.cmds);  
    br_opt(&mut irb.cmds);  
    one_opt(&mut irb.cmds);
    mem_opt(&mut irb.cmds);      // should be around last; doesn't delete any cmds
    
    // memory opts past this point has to account for usage of variables more than once.
    opeq_opt(&mut irb.cmds);     
    
    // Loop were we apply different prox optimizations and see what has the highest score out of all the optimizations
    // "Aggressive" optimization. Applies every optimization until we can't improve anymore
    let mut prev_max_val: Option<usize> = None;
    loop {
        let mut prox_rev_copy = irb.cmds.clone();
        prox_rev_opt(&mut prox_rev_copy);
        let prox_rev_score = get_score(&prox_rev_copy);

        let mut prox_copy = irb.cmds.clone();
        prox_opt(&mut prox_copy);     // changing order of this results in different programs, but similar memory opts
        let prox_score = get_score(&prox_copy);
        
        let mut chain_copy = irb.cmds.clone();
        chain_opt(&mut chain_copy);    
        let chain_score = get_score(&chain_copy);

        let max_val = prox_rev_score.max(prox_score).max(chain_score);
        if prev_max_val.is_some_and(|f| f >= max_val) { break; } // if best max val opts are worse or equal to prev, stop the loop

        if max_val == chain_score { irb.cmds = chain_copy; }
        else if max_val == prox_score { irb.cmds = prox_copy; }
        else if max_val == prox_rev_score { irb.cmds = prox_rev_copy; }
        
        prev_max_val = Some(max_val);
    }
    // prox_rev_opt(&mut irb.cmds);
    // aggr_spatial_opt(&mut irb.cmds);

    dep_opt(&mut irb.cmds);  // we may have deleted a few things, so it doesn't hurt to delete things that are not referenced again

    drop(guard);
}