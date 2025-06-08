use std::collections::HashMap;
use indexmap::IndexMap;
use crate::{core::ret_dep_list, ir::optimizations::helper::{replace_ref, replace_res_cmd}, IRCmds};

use super::helper::{ir_to_dep, ir_to_res};

pub fn mem_opt (cmds: &mut IndexMap<String, Vec<IRCmds>>) {
    let dep_list = ret_dep_list();

    let block_names: Vec<String> = cmds.iter().map(|(f, _)| f.clone()).collect();
    let mut block_name_idx = 0;
    let mut cmd_idx = 0;
    loop {
        // ===================== get metadata about cmds ============
        let mut res_to_block: HashMap<String, String> = HashMap::new();            // block tracker
        let mut res_ref_location: HashMap<String, Vec<(String, usize)>> = HashMap::new();     // reference tracker
        let mut res_location: HashMap<String, (String, IRCmds)> = HashMap::new();  // location tracker

        for (block_name, b_cmds) in cmds.iter() {
            for (idx, cmd) in b_cmds.iter().enumerate() {
                let deps = ir_to_dep(cmd.clone());
                let res = ir_to_res(cmd.clone());

                if let Some(result) = res {
                    if !deps.contains(&result) {
                        res_to_block.insert(result.clone(), block_name.clone());
                        res_ref_location.insert(result.clone(), vec![]); 
                        res_location.insert(result.clone(), (block_name.clone(), cmd.clone()));
                    }
                }

                for d in deps {
                    res_ref_location
                        .entry(d)
                        .and_modify(|x| x.push((block_name.clone(), idx)));
                }
            }
        }

        // =================== Get current block ================
        let block_name = block_names.get(block_name_idx).unwrap();
        let cmd = cmds.get_mut(block_name).unwrap().get_mut(cmd_idx).unwrap();

        // =================== Get current cmd is viable for replacement ================
        let mut replace_var: Option<(String, String)> = None;
        let deps = ir_to_dep(cmd.clone());
        let res = ir_to_res(cmd.clone());
        if let Some(result) = res {
            for dep in deps.iter() {
                // current command is a +=, /=, -=, etc.
                if deps.contains(&result) { break; }

                // check each deps, see it's declared within same block
                let bl_dep = res_to_block.get(dep).unwrap();
                if bl_dep != block_name { continue; }
                
                // check whether at deps there's no more than 1 reference 
                let nref_dep = res_ref_location.get(dep).unwrap();
                let nref_dep: Vec<usize> = nref_dep
                    .iter()
                    .filter(|&(bl, idx)| (bl != block_name) || (*idx > cmd_idx))
                    .map(|f| f.1)
                    .collect();
                if nref_dep.len() > 0 || dep_list.contains(dep) { continue; }

                // check whether result and dep and not the same (ex: l = e * l after first round of opimization)
                if result == *dep || dep_list.contains(&result) { continue; }

                /*
                check whether cmd is not a Concat operation
                [id] = concat([id], [id_one]) is not allowed. Concat is a zero-cost operation. 
                Due to the way that matrix tracker works (it works recursively), it's easier if we can gaurantee seperate inputs id to the result id
                */
                if let IRCmds::Concat { .. } = cmd { break; }

                // if so, set replace var
                replace_var = Some( (result, dep.clone()) );
                break;
            }
        }

        // =================== Replace ================
        if let Some( (a, b) ) = replace_var {
            // replace result
            replace_res_cmd(cmd, b.clone());

            // replace all references from a to b
            replace_ref(cmds, &a, b);
        }

        // =================== Get next command ================
        cmd_idx += 1;
        if cmd_idx == cmds.get(block_name).unwrap().len() {
            cmd_idx = 0;
            block_name_idx += 1;
        }
        if block_name_idx == block_names.len() {
            break;
        }
    }
}

/*
s = c * b
t = s.broadcast(dim=0, r=6)
u = r + t
v = n * u
w = m + v

===== can be converted to ======
s = c * b
s = s.broadcast(dim=0, r=6)
u = r + s
v = n * u
w = m + v
*/