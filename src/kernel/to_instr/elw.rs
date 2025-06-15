use crate::{
    kernel_decl::{Kernels, BinaryOp},
    trackers::{MatrixTracker, AccessType}, 
    IRCmds,
};

pub fn handle_elw<'a> (cmd: &IRCmds, instr: &mut Vec<Kernels>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::ElwMultiply { a, b, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Binary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                b: mat_tracker.get_input(b, AccessType::Global),
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: BinaryOp::Multiply,
                size: a_shape.iter().product()
            });
        },
        IRCmds::ElwAdd { a, b, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Binary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                b: mat_tracker.get_input(b, AccessType::Global),
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: BinaryOp::Add,
                size: a_shape.iter().product()
            });
        },
        IRCmds::ElwAddEq { s, o } => {
            let s_shape = mat_tracker.get_shape(s);

            // if both are constant, then just skip (this is evaluated at the constant tracker)
            if mat_tracker.get_constant(s).is_some() && mat_tracker.get_constant(o).is_some() {
                return; 
            }

            instr.push(Kernels::Binary { 
                a: mat_tracker.get_input(s, AccessType::Global), 
                b: mat_tracker.get_input(o, AccessType::Global),
                res: mat_tracker.get_res(s, AccessType::Global, s_shape),
                op: BinaryOp::Add,
                size: s_shape.iter().product()
            });
        },
        IRCmds::ElwMultiplyEq { s, o } => {
            let s_shape = mat_tracker.get_shape(s);

            // if both are constant, then just skip (this is evaluated at the constant tracker)
            if mat_tracker.get_constant(s).is_some() && mat_tracker.get_constant(o).is_some() {
                return; 
            }

            instr.push(Kernels::Binary { 
                a: mat_tracker.get_input(s, AccessType::Global), 
                b: mat_tracker.get_input(o, AccessType::Global),
                res: mat_tracker.get_res(s, AccessType::Global, s_shape),
                op: BinaryOp::Multiply,
                size: s_shape.iter().product()
            });
        },
        _ => {}
    }
}