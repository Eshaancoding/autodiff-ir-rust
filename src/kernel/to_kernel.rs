use indexmap::IndexMap;
use crate::{
    kernel_decl::{ComputeInstr, Procedure, ReduceOp}, 
    to_instr::{comparison::handle_comparison, elw::handle_elw, unary::handle_unary}, 
    IRCmds
};
use super::trackers::{AllocTracker, MatrixTracker, AccessType};
use super::indexing::*;

pub fn to_kernel (cmds: &IndexMap<String, Vec<IRCmds>>) {
    let mut alloc_tracker = AllocTracker::new();

    for (_, b_cmds) in cmds.iter() { 
        for cmd in b_cmds {
            alloc_tracker.step(cmd);
        }     
    }

    let mut mat_tracker = MatrixTracker::new(&alloc_tracker);
    let mut proc = Procedure::new();
    for (block_name, b_cmds) in cmds.iter() { 
        let mut instr: Vec<ComputeInstr> = vec![]; // instructions

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
                    instr.push(ComputeInstr::Reduce { 
                        a: mat_tracker.get_input(a, AccessType::XY), 
                        res: mat_tracker.get_mat(res, AccessType::XY),
                        op: ReduceOp::Sum,
                        size: 0
                    });
                },
                IRCmds::DotProduct { a, b, res } => {
                    let a_shape = mat_tracker.get_shape(a);
                    let b_shape = mat_tracker.get_shape(b);
                    
                    assert!(a_shape.len() == 2 && b_shape.len() == 2, "dot prod at kernel lowering has wrong dimensions");

                    // MAKE RES AND B CONTIGIOUS BEFORE DOT PROD
                    instr.push(ComputeInstr::DotProd { 
                        a: mat_tracker.get_input(a, AccessType::XY), 
                        b: mat_tracker.get_input(b, AccessType::XY), 
                        res: mat_tracker.get_mat(res, AccessType::XY),
                        batch_size: *a_shape.first().unwrap(),
                        input_size: *a_shape.last().unwrap(),
                        output_size: *b_shape.last().unwrap()                         
                    });
                },
                
                IRCmds::BR { block_id } => { instr.push(ComputeInstr::BR { block_id: block_id.clone() }); }
                IRCmds::BRE { block_id, a } => { instr.push(ComputeInstr::BRE { block_id: block_id.clone(), a: a.clone() }); }
                IRCmds::BRZ { block_id, a } => { instr.push(ComputeInstr::BRZ { block_id: block_id.clone(), a: a.clone() }); }
                IRCmds::EX => { instr.push(ComputeInstr::EX); }

                // data manipulation are handled by mat_tracker
                _ => {} 
            }

            println!("{:#?}", mat_tracker.sources);
            println!("{:#?}", mat_tracker.vars);
            println!("=========================");

            mat_tracker.step(cmd); 
        }
        proc.add_block(block_name, instr);
    }

    // after every command is done, make sure all the dependencies are contigious so the weights or outputs can be read

    // ...debug...
    println!("{}", proc);

    // ========= Kernel Fusion =========
    
}