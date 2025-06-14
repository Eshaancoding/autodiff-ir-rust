use indexmap::IndexMap;
use crate::{
    kernel_decl::{Kernels, KernelProcedure, ReduceOp}, 
    to_instr::{comparison::handle_comparison, elw::handle_elw, unary::handle_unary}, 
    Device, 
    IRCmds
};
use super::trackers::{
    AllocTracker, 
    MatrixTracker, 
    AccessType
};

pub fn to_kernel (device: &dyn Device, cmds: &IndexMap<String, Vec<IRCmds>>) {
    let mut alloc_tracker = AllocTracker::new(device);

    for (_, b_cmds) in cmds.iter() { 
        for cmd in b_cmds {
            alloc_tracker.step(cmd);
        }     
    }

    println!("{}", alloc_tracker);

    let mut mat_tracker = MatrixTracker::new(&alloc_tracker);
    let mut proc = KernelProcedure::new();
    for (block_name, b_cmds) in cmds.iter() { 
        let mut instr: Vec<Kernels> = vec![]; // instructions

        // TODO: order matters no? can't just do matrix tracker on random iteration
        // unless matrix tracker is dependent on per block basis, which is not.
        // this shouldn't work...
        for cmd in b_cmds {
            // instr --> kernels 
            handle_elw(cmd, &mut instr, &mat_tracker);
            handle_comparison(cmd, &mut instr, &mat_tracker);
            handle_unary(cmd, &mut instr, &mat_tracker);

            // other instr --> kernels
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
                    
                    assert!(a_shape.len() == 2 && b_shape.len() == 2, "dot prod at kernel lowering has wrong dimensions");

                    // Dot product instruction
                    instr.push(Kernels::DotProd { 
                        a: mat_tracker.get_input(a, AccessType::XY), 
                        b: mat_tracker.get_input(b, AccessType::XY), 
                        res: mat_tracker.get_res(res, AccessType::XY, &res_shape),
                        batch_size: *res_shape.first().unwrap(),
                        input_size: *a_shape.last().unwrap(),
                        output_size: *res_shape.last().unwrap()                         
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
                
                // add declaration for while and match

                // IRCmds::BR { block_id } => { instr.push(Kernels::BR { block_id: block_id.clone() }); }
                // IRCmds::BRE { block_id, a } => { instr.push(Kernels::BRE { block_id: block_id.clone(), a: a.clone() }); }
                // IRCmds::BRZ { block_id, a } => { instr.push(Kernels::BRZ { block_id: block_id.clone(), a: a.clone() }); }

                IRCmds::EX => { instr.push(Kernels::EX); }

                // data manipulation are handled by mat_tracker
                _ => {} 
            }

            // println!("{:#?}", mat_tracker.sources);
            // println!("{:#?}", mat_tracker.vars);
            // println!("=========================");

            mat_tracker.step(device, cmd); 
        }
        proc.add_block(block_name, instr);
    }

    // after every command is done, make sure all the dependencies are contigious so the weights or outputs can be read
    // ^^^^ do this at the IR Level

    // Remove movement kernel if not needed 
    // We already know it's contigious when 1. both matrixes 2. same id 3. same access expression.

    // ...debug...
    println!("{}", proc);

    // ========= Kernel Fusion =========
    
}