use crate::{kernel_decl::{UnaryOp, ComputeInstr}, trackers::MatrixTracker, IRCmds};

pub fn handle_comparison<'a> (cmd: &IRCmds, instr: &mut Vec<ComputeInstr>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::EqualZero { a, res } => {
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a), 
                res: mat_tracker.get_mat(res), 
                op: UnaryOp::EqualZero
            });
        },
        IRCmds::MoreZero { a, res } => {
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a), 
                res: mat_tracker.get_mat(res), 
                op: UnaryOp::MoreZero
            });
        },
        IRCmds::LessZero { a, res } => {
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a), 
                res: mat_tracker.get_mat(res), 
                op: UnaryOp::LessZero
            });
        },
        _ => {}
    }
}