use crate::{
    kernel_decl::{Kernels, BinaryOp},
    trackers::{KernelTracker, AccessType}, 
    IRCmds,
};

pub fn to_elw (cmd: &IRCmds, instr: &mut Vec<Kernels>, mat_tracker: &KernelTracker, kernel_id: &mut usize) {
    match cmd {
        IRCmds::ElwMultiply { a, b, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Binary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                b: mat_tracker.get_input(b, AccessType::Global),
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: BinaryOp::Multiply,
                size: a_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1;
        },
        IRCmds::ElwAdd { a, b, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Binary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                b: mat_tracker.get_input(b, AccessType::Global),
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: BinaryOp::Add,
                size: a_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1;
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
                size: s_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1;
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
                size: s_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1;
        },
        _ => {}
    }
}