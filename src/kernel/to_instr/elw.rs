use crate::{kernel_decl::{BinaryOp, ComputeInstr}, trackers::MatrixTracker, IRCmds};

pub fn handle_elw<'a> (cmd: &IRCmds, instr: &mut Vec<ComputeInstr>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::ElwMultiply { a, b, res } => {
            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(a), 
                b: mat_tracker.get_input(b),
                res: mat_tracker.get_mat(res),
                op: BinaryOp::Multiply
            });
        },
        IRCmds::ElwAdd { a, b, res } => {
            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(a), 
                b: mat_tracker.get_input(b),
                res: mat_tracker.get_mat(res),
                op: BinaryOp::Add
            });
        },
        IRCmds::ElwAddEq { s, o } => {
            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(s), 
                b: mat_tracker.get_input(o),
                res: mat_tracker.get_mat(s),
                op: BinaryOp::Add
            });
        },
        IRCmds::ElwMultiplyEq { s, o } => {
            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(s), 
                b: mat_tracker.get_input(o),
                res: mat_tracker.get_mat(s),
                op: BinaryOp::Multiply
            });
        },
        _ => {}
    }
}