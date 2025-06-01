use crate::{
    kernel_decl::{UnaryOp, ComputeInstr}, 
    trackers::MatrixTracker, 
    IRCmds,
    access_expr::AccessType
};

pub fn handle_comparison<'a> (cmd: &IRCmds, instr: &mut Vec<ComputeInstr>, mat_tracker: &MatrixTracker<'a>) {
    match cmd {
        IRCmds::EqualZero { a, res } => {
            let a_shape = mat_tracker.get_shape(a);
            let res_shape = mat_tracker.get_shape(res);
            assert!(a_shape == res_shape, "IR shape are not equal");

            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global), 
                op: UnaryOp::EqualZero,
                size: a_shape.iter().product()
            });
        },
        IRCmds::MoreZero { a, res } => {
            let a_shape = mat_tracker.get_shape(a);
            let res_shape = mat_tracker.get_shape(res);
            assert!(a_shape == res_shape, "IR shape are not equal");

            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global), 
                op: UnaryOp::MoreZero,
                size: a_shape.iter().product()
            });
        },
        IRCmds::LessZero { a, res } => {
            let a_shape = mat_tracker.get_shape(a);
            let res_shape = mat_tracker.get_shape(res);
            assert!(a_shape == res_shape, "IR shape are not equal");

            instr.push(ComputeInstr::Unary { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_mat(res, AccessType::Global), 
                op: UnaryOp::LessZero,
                size: a_shape.iter().product()
            });
        },
        _ => {}
    }
}