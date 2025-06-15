use crate::{
    kernel_decl::{Kernels, UnaryOp}, 
    trackers::{AccessType, MatrixTracker}, 
    IRCmds,
};

pub fn handle_comparison<'a> (cmd: &IRCmds, instr: &mut Vec<Kernels>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::EqualZero { a, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape), 
                op: UnaryOp::EqualZero,
                size: a_shape.iter().product()
            });
        },
        IRCmds::MoreZero { a, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape), 
                op: UnaryOp::MoreZero,
                size: a_shape.iter().product()
            });
        },
        IRCmds::LessZero { a, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape), 
                op: UnaryOp::LessZero,
                size: a_shape.iter().product()
            });
        },
        _ => {}
    }
}

// I should really use a macro for this