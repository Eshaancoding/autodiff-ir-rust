use crate::{
    conflicts::insert_alloc, 
    core::ret_dep_list, 
    kernel::conflicts::fix_access_conflicts, 
    kernel_decl::{KernelProcedure, Kernels}, to_instr::{to_comp, to_control, to_elw, to_special, to_unary}, 
    trackers::Location, 
    Device, 
    IRProcedure
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
    // Initialize + step through ir procedure for allocation tracker
    let dep_vars = ret_dep_list();
    let mut kernel_tracker = KernelTracker::new(&dep_vars);
    let mut kernel_proc = convert_to_proc(device, &mut kernel_tracker, proc);

    // ========= Kernel Fixes =========
    insert_alloc(&mut kernel_proc, &kernel_tracker.alloc_tracker);
    fix_access_conflicts(&mut kernel_proc);

    // ========= Kernel Optimizations =========

    // ========= Kernel Fusion =========
    // go to first OpenCL implementation --> implement --> then maybe reach back to x86 and see if you can make that better

    // ========= Kernel Tuning =========


    // ========= Return =========

    // ...debug...
    println!("{}", kernel_tracker.alloc_tracker);
    println!("{}", kernel_proc);

    kernel_proc
}