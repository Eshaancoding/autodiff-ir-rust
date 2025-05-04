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
    alloc_tracker.print();

    let mut idx_tracker = MatrixTracker::new(&alloc_tracker);
    for (_, b_cmds) in cmds.iter() { 
        for cmd in b_cmds {
            idx_tracker.step(cmd);
        }
        idx_tracker.print();
        break;
    }
}