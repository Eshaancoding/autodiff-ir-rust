use crate::{
    kernel_decl::{ComputeInstr, BinaryOp},
    trackers::{MatrixTracker, AccessType}, 
    IRCmds,
};

pub fn handle_elw<'a> (cmd: &IRCmds, instr: &mut Vec<ComputeInstr>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::ElwMultiply { a, b, res } => {
            let a_shape = mat_tracker.get_shape(a);
            let b_shape = mat_tracker.get_shape(b);
            assert!(a_shape == b_shape, "IR shape are not equal");

            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                b: mat_tracker.get_input(b, AccessType::Global),
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: BinaryOp::Multiply,
                size: a_shape.iter().product()
            });
        },
        IRCmds::ElwAdd { a, b, res } => {
            let a_shape = mat_tracker.get_shape(a);
            let b_shape = mat_tracker.get_shape(b);
            assert!(a_shape == b_shape, "IR shape are not equal");

            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                b: mat_tracker.get_input(b, AccessType::Global),
                res: mat_tracker.get_res(res, AccessType::Global, a_shape),
                op: BinaryOp::Add,
                size: a_shape.iter().product()
            });
        },
        IRCmds::ElwAddEq { s, o } => {
            let s_shape = mat_tracker.get_shape(s);
            let o_shape = mat_tracker.get_shape(o);
            assert_eq!(s_shape, o_shape, "IR shape are not equal");

            instr.push(ComputeInstr::Binary { 
                a: mat_tracker.get_input(s, AccessType::Global), 
                b: mat_tracker.get_input(o, AccessType::Global),
                res: mat_tracker.get_res(s, AccessType::Global, s_shape),
                op: BinaryOp::Add,
                size: s_shape.iter().product()
            });
        },
        IRCmds::ElwMultiplyEq { s, o } => {
            let s_shape = mat_tracker.get_shape(s);
            let o_shape = mat_tracker.get_shape(o);
            assert!(s_shape == o_shape, "IR shape are not equal");

            instr.push(ComputeInstr::Binary { 
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