use crate::{
    kernel_decl::{ComputeInstr, BinaryOp},
    trackers::MatrixTracker, 
    IRCmds,
    access_expr::AccessType
};

pub fn handle_elw<'a> (cmd: &IRCmds, instr: &mut Vec<ComputeInstr>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::ElwMultiply { a, b, res } => {
            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                b: mat_tracker.get_input(b, AccessType::Global),
                res: mat_tracker.get_mat(res, AccessType::Global),
                op: BinaryOp::Multiply
            });
        },
        IRCmds::ElwAdd { a, b, res } => {
            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                b: mat_tracker.get_input(b, AccessType::Global),
                res: mat_tracker.get_mat(res, AccessType::Global),
                op: BinaryOp::Add
            });
        },
        IRCmds::ElwAddEq { s, o } => {
            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(s, AccessType::Global), 
                b: mat_tracker.get_input(o, AccessType::Global),
                res: mat_tracker.get_mat(s, AccessType::Global),
                op: BinaryOp::Add
            });
        },
        IRCmds::ElwMultiplyEq { s, o } => {
            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(s, AccessType::Global), 
                b: mat_tracker.get_input(o, AccessType::Global),
                res: mat_tracker.get_mat(s, AccessType::Global),
                op: BinaryOp::Multiply
            });
        },
        _ => {}
    }
}