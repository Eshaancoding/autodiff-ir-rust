use crate::{
    alloc::{alloc_out_fused, alloc_switch, alloc_temp_opt, insert_alloc},  
    fusion::{dp_elw::fuse_dp_expr, fuse_elw_expr, fuse_rd_expr}, 
    kernel_decl::{KernelProcedure, Kernels}, 
    memory::{get_score, mem_opt, prox_opt, prox_rev_opt}, 
    to_instr::{to_comp, to_control, to_elw, to_special, to_unary}, 
    Device, 
    IRProcedure
};
use super::trackers::KernelTracker;

pub fn convert_to_proc (device: &dyn Device, kernel_tracker: &mut KernelTracker, proc: &IRProcedure) -> KernelProcedure {
    let mut kernels: Vec<Kernels> = vec![];

    for cmd in proc.iter() {
        to_elw(cmd, &mut kernels, &kernel_tracker);
        to_comp(cmd, &mut kernels, &kernel_tracker);
        to_unary(cmd, &mut kernels, &kernel_tracker);
        to_special(device, cmd, &mut kernels, &kernel_tracker);

        to_control(device, cmd, &mut kernels, kernel_tracker); // note: recursive

        // step kernel tracker
        kernel_tracker.step(device, cmd);
    }

    KernelProcedure::new(
        kernels,
        proc.id.clone()
    )
}

pub fn to_kernel (device: &dyn Device, proc: &IRProcedure) -> KernelProcedure {

    // ========== Create initial procedure with kernel tracker ========== 
    let mut kernel_tracker = KernelTracker::new();
    let mut kernel_proc = convert_to_proc(device, &mut kernel_tracker, proc);

    // ========= Memory Optimization ==========
    let var_changed = kernel_proc.get_var_changed(); 
    mem_opt(&mut kernel_proc, &var_changed);

    let mut prev_max_val: Option<usize> = None;
    loop {
        // copy, change, score
        let mut prox_rev_copy = kernel_proc.clone();
        prox_rev_opt(&mut prox_rev_copy);
        let prox_rev_score = get_score(&mut prox_rev_copy);

        // copy, change, score
        let mut prox_copy = kernel_proc.clone();
        prox_opt(&mut prox_copy);                   // TODO: Doesn't have safegaurd for swaps in between IF/While. Still okay as it pasts the tests? 
        let prox_score = get_score(&mut prox_copy);

        // Calculate max
        let max_val = prox_rev_score.max(prox_score);
        if prev_max_val.is_some_and(|f| f >= max_val) { break; } 
        // if best max val opts are worse or equal to prev, stop the loop -> achieved maximum

        // set to current proc
        if max_val == prox_score { kernel_proc = prox_copy; }
        else if max_val == prox_rev_score { kernel_proc = prox_rev_copy; }
        
        prev_max_val = Some(max_val);
    }

    // ========= Insert allocations + deallocations =========
    insert_alloc(device, &mut kernel_proc);

    // ========= Kernel Fusion =========
    fuse_elw_expr(&mut kernel_proc); 
    fuse_dp_expr(&mut kernel_proc);
    fuse_rd_expr(&mut kernel_proc);

    // ========= Allocation Optimizations =========
    alloc_switch(&mut kernel_proc);

    // you need access expression simplification even more for this to work the best
    alloc_temp_opt(&mut kernel_proc);
    
    alloc_out_fused(&mut kernel_proc);
    // lastly, tetris opt

    // ====== Kernel checks for sanity purposes ======= 
    // in fusion kernels, check if only allowed kernel irs are inserted

    // ========= Kernel Tuning =========


    // ========= Return =========

    // ...debug...
    println!("{}", kernel_proc);

    kernel_proc
}