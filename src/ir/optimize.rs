// use crate::ir_print;
use crate::{core::env_flags::disable_ir_opt,  IRB};
use crate::ir::opts::{
    const_begin, dep_opt, repeat_opt, track_var_changed
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
    // multiple views
    // multiple contigious
    //     when you do this --> everytime you use "keep" --> call contigious function
    
    drop(guard);
}