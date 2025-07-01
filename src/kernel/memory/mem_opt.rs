use std::collections::HashMap;
use crate::{core::ret_dep_list, kernel_decl::{KernelProcedure, Kernels}};

#[derive(Debug, PartialEq)]
pub struct RefLocation {
    proc_id: String,
    idx: usize
}

/*
Technically, we don't need this optimization 
The main premise of this opt is to reuse ids that are only being used once
however, we anyways globalize all ids into one at a future optimization ("tetris optimzation" I like to call it)
So why mem opt? It actually helps a lot with grouping like operations together
This massively helps kernel fusion, which is extremely important
*/
pub fn mem_opt (proc: &mut KernelProcedure, var_changed: &Vec<String>) {
    let mut dep_list = ret_dep_list();

    // ===================== Track variables that are from while or if statements ===================== 
    proc.step_cmd_fusion(&mut |proc, idx| {
        let cmd = proc.get(*idx).unwrap();

        if let Kernels::If { conditions, .. } = cmd {
            for (c, _) in conditions.iter() {
                dep_list.insert(c.clone());
            }
        }
        else if let Kernels::While { conditional_var, .. } = cmd {
            dep_list.insert(conditional_var.clone());
        }
    });

    // ========== get metadata about cmds ========== 
    let mut res_to_procid: HashMap<String, String> = HashMap::new();
    let mut res_ref_location: HashMap<String, Vec<RefLocation>> = HashMap::new();

    let mut func_track = |proc: &mut KernelProcedure, idx: &mut usize| {
        let cmd = proc.get(*idx).unwrap();

        let deps = cmd.get_dep_id();
        let res = cmd.get_res();

        if let Some(result) = res {
            if !deps.contains(&result) {
                res_to_procid.insert(result.clone(), proc.id.clone());
                res_ref_location.insert(result.clone(), vec![]); 
            }
        }

        for d in deps {
            res_ref_location
                .entry(d.clone())
                .and_modify(|x| x.push(RefLocation {
                    proc_id: proc.id.clone(),
                    idx: *idx
                }));
        }

        true
    };

    
    proc.step_cmd(&mut func_track);

    // ========== Step through Cmds ========== 
    let mut func = |proc: &mut KernelProcedure, c_idx: &mut usize| {
        let idx = *c_idx;        

        let KernelProcedure {id, kernels } = proc;
        let cmd = kernels.get_mut(idx).unwrap();
        let deps = cmd.get_dep_id();
        let res = cmd.get_res();
        let mut replace_var: Option<(String, String)> = None;

        if let Some(result) = res {
            for &dep in deps.iter() {
                // current command is a +=, /=, -=, etc.
                if deps.contains(&result) { break; }

                // if deps is in var_changed
                if var_changed.contains(result) { break; }

                // check each deps, see it's declared within same block
                let pr_dep = res_to_procid.get(dep).unwrap();
                if pr_dep != id { continue; }
                
                // check whether at deps there's no more than 1 reference 
                let nref_dep = res_ref_location.get(dep).unwrap();
                let nref_dep: Vec<usize> = nref_dep
                    .iter()
                    .filter(|&rl| (rl.proc_id != *id) || (rl.idx > idx))
                    .map(|f| f.idx)
                    .collect();

                if nref_dep.len() > 0 || dep_list.contains(dep) { continue; }

                // check whether result and dep and not the same (ex: l = e * l after first round of opimization)
                if result == dep || dep_list.contains(result) { continue; }

                // We require these 3 commands to have unique result + input IDs as well.
                // If we allow same ids, then we can get a result like this: by = sum(by, dim=-1)
                // If you look at an access expression of something like this: 
                // M (id: by, access: #x) = Sum (Vec/X: 128, Reduce/Y: 2)  (M (id: by, access: ((#y << 7) + #x)))
                // by is being accessed + written at the same time (with different access expressions), which can cause a conflict.
                // This does hurt the memory optimizations, but we have more ;)
                if let Kernels::DotProd { .. } = cmd { break; }
                if let Kernels::Reduce { .. } = cmd { break; }
                if let Kernels::Movement { .. } = cmd { break; }

                // Data manipulation 
                // if so, set replace var
                replace_var = Some( (result.clone(), dep.clone()) );
                break;
            }
        }

        // =================== Replace ================
        if let Some( (a, b) ) = replace_var {
            // replace result
            cmd.change_res_id(b.clone());

            // replace all references from a to b
            // re-iterating O(n^2) <-- could be faster since we already have the locations
            proc.replace_ref(&a, b.clone());

            // update metadata
            let v = res_to_procid.remove(&a);
            let ref_loc = res_ref_location.remove(&a);

            if let Some(v) = v {
                res_to_procid.insert(b.clone(), v);
            }

            if let Some(ref_loc) = ref_loc {
                if let Some(b_rl) = res_ref_location.get_mut(&b) {
                    b_rl.extend(ref_loc);
                }
            }
        }

        true
    };

    proc.step_cmd(&mut func);
}