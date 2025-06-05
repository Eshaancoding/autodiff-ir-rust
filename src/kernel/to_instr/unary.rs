use crate::{
    kernel_decl::{UnaryOp, ComputeInstr}, 
    trackers::{MatrixTracker, AccessType}, 
    IRCmds,
};

pub fn handle_unary<'a> (cmd: &IRCmds, instr: &mut Vec<ComputeInstr>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::Exp2 { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);

            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Exp2 ,
                size: a_shape.iter().product()
            }) 
        },
        IRCmds::Log2 { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);

            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Log2,
                size: a_shape.iter().product() 
            }) 
        },
        IRCmds::Sin { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);
            
            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Sin ,
                size: a_shape.iter().product()
            }) 
        },
        IRCmds::Recip { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);

            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Recip,
                size: a_shape.iter().product()
            }) 
        },
        IRCmds::Sqrt { a, res } => { 
            let a_shape = mat_tracker.get_shape(a);

            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: UnaryOp::Sqrt,
                size: a_shape.iter().product()
            }) 
        },
        _ => {}
    }
}