use crate::{
    conflicts::insert_alloc, core::ret_dep_list, fusion::{dp_elw::fuse_dp_expr, fuse_elw_expr, fuse_rd_expr}, kernel_decl::{KernelProcedure, Kernels}, to_instr::{to_comp, to_control, to_elw, to_special, to_unary}, trackers::Location, Device, IRProcedure
};
use super::trackers::KernelTracker;

pub fn convert_to_proc<'a> (device: &dyn Device, kernel_tracker: &mut KernelTracker<'a>, proc: &'a IRProcedure) -> KernelProcedure {
    let mut kernels: Vec<Kernels> = vec![];

    for cmd in proc.main.iter() {
        to_elw(cmd, &mut kernels, &kernel_tracker);
        to_comp(cmd, &mut kernels, &kernel_tracker);
        to_unary(cmd, &mut kernels, &kernel_tracker);
        to_special(device, cmd, &mut kernels, &kernel_tracker);

        let merge_loc = Location {
            proc_id: proc.id.clone(),
            loc: kernels.len()+1
        };
        to_control(device, cmd, &mut kernels, kernel_tracker, &merge_loc); // note: recursive

        // step kernel tracker
        kernel_tracker.step(device, cmd, Location {
            proc_id: proc.id.clone(),
            loc: kernels.len()
        });
    }

    KernelProcedure::new(
        kernels,
        proc.id.clone()
    )
}

pub fn to_kernel (device: &dyn Device, proc: &IRProcedure) -> KernelProcedure {

    // ========== Create initial procedure with kernel tracker ========== 
    let dep_vars = ret_dep_list();
    let mut kernel_tracker = KernelTracker::new(&dep_vars);
    let mut kernel_proc = convert_to_proc(device, &mut kernel_tracker, proc);

    // ========= Insert allocations + deallocations =========
    insert_alloc(&mut kernel_proc, &kernel_tracker.alloc_tracker);

    // ========= Kernel Fusion =========
    fuse_elw_expr(&mut kernel_proc); 
    fuse_dp_expr(&mut kernel_proc);
    fuse_rd_expr(&mut kernel_proc);

    // ========= Kernel Optimizations =========

    // ====== Kernel checks for sanity purposes ======= 

    // ========= Kernel Tuning =========


    // ========= Return =========

    // ...debug...
    println!("{}", kernel_proc);

    kernel_proc
}