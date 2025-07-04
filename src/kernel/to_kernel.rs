use crate::{
    alloc::{alloc_in, alloc_out_fused, alloc_switch, alloc_temp_opt, insert_alloc, tetris_opt}, 
    fusion::{dp_elw::fuse_dp_expr, fuse_elw_expr, fuse_rd_expr}, 
    helper::simplify_expr::simplify_global_expr, 
    kernel_decl::{KernelProcedure, Kernels}, 
    memory::{get_score, mem_opt, prox_opt, prox_rev_opt}, 
    to_instr::{to_comp, to_control, to_elw, to_special, to_unary}, 
    Device, 
    IRProcedure
};
use super::trackers::KernelTracker;

pub fn convert_to_proc (device: &dyn Device, kernel_tracker: &mut KernelTracker, proc: &IRProcedure, kernel_id: &mut usize) -> KernelProcedure {
    let mut kernels: Vec<Kernels> = vec![];

    for cmd in proc.iter() {
        to_elw(cmd, &mut kernels, &kernel_tracker, kernel_id);
        to_comp(cmd, &mut kernels, &kernel_tracker, kernel_id);
        to_unary(cmd, &mut kernels, &kernel_tracker, kernel_id);
        to_special(device, cmd, &mut kernels, &kernel_tracker, kernel_id);

        to_control(device, cmd, &mut kernels, kernel_tracker, kernel_id); // note: recursive

        // step kernel tracker
        kernel_tracker.step(device, cmd);
    }

    KernelProcedure::new(
        kernels,
        proc.id.clone()
    )
}

pub fn to_kernel (device: &dyn Device, proc: &IRProcedure) -> (KernelProcedure, KernelTracker) {
    let mut kernel_id: usize = 0;

    // ========== Create initial procedure with kernel tracker ========== 
    let mut kernel_tracker = KernelTracker::new();
    let mut kernel_proc = convert_to_proc(device, &mut kernel_tracker, proc, &mut kernel_id);

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
        // TODO: Doesn't have safegaurd for swaps in between IF/While. Still okay as it pasts the tests? 
        let mut prox_copy = kernel_proc.clone();
        prox_opt(&mut prox_copy);                   
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
    insert_alloc(device, &mut kernel_proc, &var_changed);

    // ========= Kernel Fusion =========
    fuse_elw_expr(&mut kernel_proc, &mut kernel_id); 
    fuse_dp_expr(&mut kernel_proc, &mut kernel_id);
    fuse_rd_expr(&mut kernel_proc, &mut kernel_id);

    // ============== Allocation Optimizations ==================
    // ideally put insert alloc here plz 
    simplify_global_expr(&mut kernel_proc); // good preq for alloc_temp_opt (need same global params)
    
    // Try to match allocs/deallocs inside fusion commands
    alloc_in(&mut kernel_proc);
    alloc_switch(&mut kernel_proc); // good preq 
    
    // if allocs + dealloc in same fusion --> replace with temporary
    alloc_temp_opt(&mut kernel_proc);  // only supports reduce + dot prod... any chance for ELW too (multisized temp)

    // Collate all temporary memory into one --> tetris opt 
    if false {
        tetris_opt(&mut kernel_proc, &var_changed);
    }
    // ^^^^^^^^^^^ There's a massive error with this ^^^^^^^^^^^ 
    
    // remove the allocs from fusion (fusion runtime cannot handle allocs or deallocs)
    alloc_out_fused(&mut kernel_proc);

    // ====== Kernel checks for sanity purposes ======= 
    // in fusion kernels, check if only allowed kernel irs are inserted
    // also check if first kernel is a dotprod/reduce

    // ========= Kernel Tuning =========


    // ========= Return =========
    (kernel_proc, kernel_tracker)
}