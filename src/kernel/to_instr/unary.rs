use crate::{kernel_decl::{UnaryOp, ComputeInstr}, trackers::MatrixTracker, IRCmds};

pub fn handle_unary<'a> (cmd: &IRCmds, instr: &mut Vec<ComputeInstr>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::Exp2 { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a), 
                res: mat_tracker.get_mat(res),
                op: UnaryOp::Exp2 
            }) 
        },
        IRCmds::Log2 { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a), 
                res: mat_tracker.get_mat(res),
                op: UnaryOp::Log2 
            }) 
        },
        IRCmds::Sin { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a), 
                res: mat_tracker.get_mat(res),
                op: UnaryOp::Sin 
            }) 
        },
        IRCmds::Neg { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a), 
                res: mat_tracker.get_mat(res),
                op: UnaryOp::Neg
            }) 
        },
        IRCmds::Recip { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a), 
                res: mat_tracker.get_mat(res),
                op: UnaryOp::Recip
            }) 
        },
        IRCmds::Sqrt { a, res } => { 
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a), 
                res: mat_tracker.get_mat(res),
                op: UnaryOp::Sqrt
            }) 
        },
        _ => {}
    }
}