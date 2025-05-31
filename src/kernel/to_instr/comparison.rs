use crate::{
    kernel_decl::{UnaryOp, ComputeInstr}, 
    trackers::MatrixTracker, 
    IRCmds,
    access_expr::AccessType
};

pub fn handle_comparison<'a> (cmd: &IRCmds, instr: &mut Vec<ComputeInstr>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::EqualZero { a, res } => {
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global), 
                op: UnaryOp::EqualZero
            });
        },
        IRCmds::MoreZero { a, res } => {
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global), 
                op: UnaryOp::MoreZero
            });
        },
        IRCmds::LessZero { a, res } => {
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global), 
                op: UnaryOp::LessZero
            });
        },
        _ => {}
    }
}