use crate::{
    kernel_decl::{KernelProcedure, Kernels}, to_kernel::convert_to_proc, trackers::{KernelTracker, Location}, Device, IRCmds, IRProcedure
};

fn handle_embed_proc (device: &dyn Device, mat_tracker: &mut KernelTracker, p: &IRProcedure) -> KernelProcedure {
    // clone matrix tracker
    let mut mat_tracker_copy = mat_tracker.clone(); // EXPENSIVE OPERATION! Can be optimized?
    
    // convert proc
    let p = convert_to_proc(device, &mut mat_tracker_copy, p);

    // merge copy matrix tracker to orig matrix tracker (or any other variables if needed) 
    // mat_tracker.merge(mat_tracker_copy, merge_loc); 

    p
}

pub fn to_control (device: &dyn Device, cmd: &IRCmds, instr: &mut Vec<Kernels>, mat_tracker: &mut KernelTracker) {
    match cmd {
        IRCmds::If { conditions, else_proc } => {
            let conditions: Vec<(String, KernelProcedure)> = conditions.iter().map(|(cond, p)| {
                (cond.clone(), handle_embed_proc(device, mat_tracker, p))
            }).collect();

            let else_proc = else_proc.as_ref().map(|p| {
                handle_embed_proc(device, mat_tracker, p) 
            });

            instr.push(Kernels::If {
                conditions,
                else_proc 
            });
        },
        IRCmds::While { conditional_var, block } => {
            instr.push(Kernels::While { 
                conditional_var: conditional_var.clone(), 
                block: handle_embed_proc(device, mat_tracker, block) 
            });
        }
        
        // add declaration for while and match
        IRCmds::EX => { 
            instr.push(Kernels::EX); 
        }
        _ => {}
    }
}

