use std::collections::HashMap;

use indexmap::IndexMap;
use crate::{
    access_expr::AccessType, helper::debug::console_list, kernel_decl::{ComputeInstr, ReduceOp}, to_instr::{comparison::handle_comparison, elw::handle_elw, unary::handle_unary}, IRCmds
};
use super::{helper::debug::console_hashmap, trackers::{AllocTracker, MatrixTracker}};

pub fn to_kernel (cmds: &IndexMap<String, Vec<IRCmds>>) {
    let mut alloc_tracker = AllocTracker::new();

    for (_, b_cmds) in cmds.iter() { 
        for cmd in b_cmds {
            alloc_tracker.step(cmd);
        }     
    }

    let mut mat_tracker = MatrixTracker::new(&alloc_tracker);
    let mut cmd_computeinstr: HashMap<String, Vec<ComputeInstr>> = HashMap::new();
    for (block_name, b_cmds) in cmds.iter() { 
        let mut instr: Vec<ComputeInstr> = vec![]; // instructions
        for cmd in b_cmds {
            mat_tracker.step(cmd); 

            // instr --> kernels 
            handle_elw(cmd, &mut instr, &mat_tracker);
            handle_comparison(cmd, &mut instr, &mat_tracker);
            handle_unary(cmd, &mut instr, &mat_tracker);

            // other instr --> kernels
            match cmd {
                IRCmds::Sum { a, dim, res } => {
                    // ===== EXPR GEN NEED TO CONSIDER DIM IN SUM ====
                    // MAKE CONTIGIOUS BEFORE REDUCE
                    instr.push(ComputeInstr::Reduce { 
                        a: mat_tracker.get_input(a, AccessType::Global), 
                        res: mat_tracker.get_mat(res, AccessType::Global),
                        op: ReduceOp::Sum
                    });
                },
                IRCmds::DotProduct { a, b, res } => {
                    // MAKE RES AND B CONTIGIOUS BEFORE DOT PROD
                    instr.push(ComputeInstr::DotProd { 
                        a: mat_tracker.get_input(a, AccessType::XY), 
                        b: mat_tracker.get_input(b, AccessType::XY), 
                        res: mat_tracker.get_mat(res, AccessType::XY) 
                    });
                },
                
                IRCmds::BR { block_id } => { instr.push(ComputeInstr::BR { block_id: block_id.clone() }); }
                IRCmds::BRE { block_id, a } => { instr.push(ComputeInstr::BRE { block_id: block_id.clone(), a: a.clone() }); }
                IRCmds::BRZ { block_id, a } => { instr.push(ComputeInstr::BRZ { block_id: block_id.clone(), a: a.clone() }); }
                IRCmds::EX => { instr.push(ComputeInstr::EX); }

                // data manipulation are handled by mat_tracker
                _ => {} 
            }
        }
        cmd_computeinstr.insert(block_name.clone(), instr);
    }

    // ...debug...
    println!("{:#?}", cmd_computeinstr);
    console_list(cmd_computeinstr.get("main").unwrap()); // only available in run()

    // ========= Kernel Fusion =========
    
}