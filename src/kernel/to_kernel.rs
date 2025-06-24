use crate::{
    kernel::opt::fix_access_conflicts, kernel_decl::{KernelProcedure, Kernels}, to_instr::{to_comp, to_control, to_elw, to_special, to_unary}, Device, IRProcedure
};
use super::trackers::KernelTracker;

pub fn convert_to_proc (device: &dyn Device, mat_tracker: &mut KernelTracker, proc: &IRProcedure) -> KernelProcedure {
    let mut kernels: Vec<Kernels> = vec![];

    for cmd in proc.main.iter() {
        to_elw(cmd, &mut kernels, &mat_tracker);
        to_comp(cmd, &mut kernels, &mat_tracker);
        to_unary(cmd, &mut kernels, &mat_tracker);
        to_special(device, cmd, &mut kernels, &mat_tracker);
        to_control(device, cmd, &mut kernels, &mat_tracker); // note: recursive

        // step matrix tracker
        mat_tracker.step(device, cmd);
    }

    KernelProcedure::new(
        kernels,
        proc.id.clone()
    )
}

pub fn to_kernel (device: &dyn Device, proc: &IRProcedure) -> KernelProcedure {
    // Initialize + step through ir procedure for allocation tracker
    let mut kernel_tracker = KernelTracker::new();
    let mut kernel_proc = convert_to_proc(device, &mut kernel_tracker, proc);

    // ...debug...
    println!("{}", kernel_proc);

    // ========= Kernel Optimizations =========
    fix_access_conflicts(&mut kernel_proc); // not necessarily an "optimization". Program wouldn't run correctly without this

    // ========= Kernel Fusion =========
    // go to first OpenCL implementation --> implement --> then maybe reach back to x86 and see if you can make that better

    // ========= Kernel Tuning =========


    // ========= Return =========

    kernel_proc
}