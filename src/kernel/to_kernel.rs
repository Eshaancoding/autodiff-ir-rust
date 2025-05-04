use indexmap::IndexMap;
use crate::IRCmds;
use super::trackers::{AllocTracker, MatrixTracker};

pub fn to_kernel (cmds: &IndexMap<String, Vec<IRCmds>>) {
    let mut alloc_tracker = AllocTracker::new();

    for (_, b_cmds) in cmds.iter() { 
        for cmd in b_cmds {
            alloc_tracker.step(cmd);
        }     
    }

    let mut mat_tracker = MatrixTracker::new(&alloc_tracker);
    for (_, b_cmds) in cmds.iter() { 
        for cmd in b_cmds {
            mat_tracker.step(cmd);
        }
    }

    mat_tracker.print_alloc_tracker();

    // DEBUG MAT TRACKER AND ACCESS EXPR
    let var_name = "bn";
    mat_tracker.print_raw(&var_name.to_string());
    println!("{}:\n\t{}", var_name, mat_tracker.get_input(var_name.to_string()));
}