use crate::{
    kernel_decl::{UnaryOp, Kernels}, 
    trackers::{KernelTracker, AccessType}, 
    IRCmds,
};

pub fn to_unary (cmd: &IRCmds, instr: &mut Vec<Kernels>, mat_tracker: &KernelTracker, kernel_id: &mut usize) {
    match cmd {
        IRCmds::Exp2 { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Exp2 ,
                size: a_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1;
        },
        IRCmds::Log2 { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Log2,
                size: a_shape.iter().product(),
                id: *kernel_id 
            });

            *kernel_id += 1;
        },
        IRCmds::Sin { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);
            
            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Sin ,
                size: a_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1;
        },
        IRCmds::Recip { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Recip,
                size: a_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1;
        },
        IRCmds::Sqrt { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Sqrt,
                size: a_shape.iter().product(),
                id: *kernel_id
            });

            *kernel_id += 1; 
        },
        _ => {}
    }
}