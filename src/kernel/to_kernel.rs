use crate::{
    kernel_decl::{KernelProcedure, Kernels}, 
    to_instr::{to_comp, to_control, to_elw, to_special, to_unary}, 
    Device, 
    IRProcedure
};
use super::trackers::MatrixTracker;

pub fn convert_to_proc (device: &dyn Device, mat_tracker: &mut MatrixTracker, proc: &IRProcedure) -> KernelProcedure {
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
    let mut mat_tracker = MatrixTracker::new();
    let kernel_proc = convert_to_proc(device, &mut mat_tracker, proc);

    // ...debug...
    println!("{}", kernel_proc);

    // ========= Kernel Fusion =========

    // ========= Kernel Tuning =========

    // ========= Return =========

    kernel_proc
}