use crate::{
    kernel_decl::{UnaryOp, ComputeInstr}, 
    trackers::MatrixTracker, 
    IRCmds,
    access_expr::AccessType 
};

pub fn handle_unary<'a> (cmd: &IRCmds, instr: &mut Vec<ComputeInstr>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::Exp2 { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global),
                op: UnaryOp::Exp2 
            }) 
        },
        IRCmds::Log2 { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global),
                op: UnaryOp::Log2 
            }) 
        },
        IRCmds::Sin { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global),
                op: UnaryOp::Sin 
            }) 
        },
        IRCmds::Recip { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global),
                op: UnaryOp::Recip
            }) 
        },
        IRCmds::Sqrt { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global),
                op: UnaryOp::Sqrt
            }) 
        },
        _ => {}
    }
}