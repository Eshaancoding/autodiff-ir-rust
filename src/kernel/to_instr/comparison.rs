use crate::{
    kernel_decl::{Kernels, UnaryOp}, 
    trackers::{AccessType, KernelTracker}, 
    IRCmds,
};

pub fn to_comp (cmd: &IRCmds, instr: &mut Vec<Kernels>, mat_tracker: &KernelTracker, kernel_id: &mut usize) {
    match cmd {
        IRCmds::EqualZero { a, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape), 
                op: UnaryOp::EqualZero,
                size: a_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1;
        },
        IRCmds::MoreZero { a, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape), 
                op: UnaryOp::MoreZero,
                size: a_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1;
        },
        IRCmds::LessZero { a, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape), 
                op: UnaryOp::LessZero,
                size: a_shape.iter().product(),
                id: *kernel_id
            });
            
            *kernel_id += 1;
        },
        _ => {}
    }
}

// I should really use a macro for this