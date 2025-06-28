// Dot product, movement, and sum are considered "special" kernels
// The reason why is because they require different access expressions than most other kernels
// If you notice at mem_opt, we skip these special kernels because their access expressions can be intrude one another
// This is not the case for Element-wise, comparison, or unary operations

use crate::{
    kernel_decl::{Kernels, ReduceOp},
    trackers::{KernelTracker, AccessType}, 
    IRCmds,
    Device
};

pub fn to_special (device: &dyn Device, cmd: &IRCmds, instr: &mut Vec<Kernels>, mat_tracker: &KernelTracker) {
    match cmd {
        IRCmds::Sum { a, res } => {
            let mut exp_dim = mat_tracker.get_shape(a).clone();
            let vec_size = exp_dim.first().unwrap().clone();
            let reduce_size = exp_dim.last().unwrap().clone();
            exp_dim.remove(exp_dim.len()-1); // remove last dim

            instr.push(Kernels::Reduce { 
                a: mat_tracker.get_input(a, AccessType::XY),
                res: mat_tracker.get_res(res, AccessType::XY, &exp_dim),
                op: ReduceOp::Sum,
                vec_size,
                reduce_size
            });
        },
        IRCmds::DotProduct { a, b, res } => {
            let a_shape = mat_tracker.get_shape(a);  
            let b_shape = mat_tracker.get_shape(b); 
            let res_shape = device.dot_prod_shape(a_shape, b_shape);
            
            let convert_tuple = |v: &Vec<usize>| { 
                assert_eq!(v.len(), 2, "convert tuple to length 2 invalid");
                (v[0], v[1])
            };
            
            assert!(a_shape.len() == 2 && b_shape.len() == 2, "dot prod at kernel lowering has wrong dimensions");

            // Dot product instruction
            instr.push(Kernels::DotProd { 
                a: mat_tracker.get_input(a, AccessType::XY), 
                b: mat_tracker.get_input(b, AccessType::XY), 
                res: mat_tracker.get_res(res, AccessType::XY, &res_shape),
                a_shape: convert_tuple(&a_shape),
                b_shape: convert_tuple(&b_shape),
                res_shape: convert_tuple(&res_shape)
            });
        },
        IRCmds::Contigious { a, res } => {
            let a_shape = mat_tracker.get_shape(a);

            instr.push(Kernels::Movement { 
                a: mat_tracker.get_input(a, AccessType::Global), 
                res: mat_tracker.get_res(res, AccessType::Global, a_shape), 
                size: a_shape.iter().product()
            });
        },

        // "special" in not the traditional sense. Device doesn't implement AllocEntry
        // this exists for allocations
        IRCmds::CreateMat { contents, dim, id } => {
            instr.push(Kernels::Alloc { 
                id: id.clone(), 
                size: dim.iter().product::<usize>(), 
                content: Some(contents.clone())
            })
        }
        _ => {}
    }
}
