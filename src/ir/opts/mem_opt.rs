use std::collections::{HashMap, HashSet};
use crate::{
    core::ret_dep_list, 
    ir::helper::{ir_to_dep, ir_to_res, replace_res_cmd, replace_ref}, 
    IRCmds, 
    IRProcedure
};

#[derive(Debug, PartialEq)]
pub struct RefLocation {
    proc_id: String,
    idx: usize
}

pub fn mem_opt (proc: &mut IRProcedure, var_changed: &Vec<String>) {
    let dep_list = ret_dep_list();

    // ========== get metadata about cmds ========== 
    let mut res_to_procid: HashMap<String, String> = HashMap::new();
    let mut res_ref_location: HashMap<String, Vec<RefLocation>> = HashMap::new();
    let mut res_constants: HashSet<String> = HashSet::new();

    let mut func_track = |proc: &mut IRProcedure, idx: &mut usize| {
        let cmd = proc.get(*idx).unwrap();

        let deps = ir_to_dep(cmd);
        let res = ir_to_res(&cmd);

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

        if let IRCmds::CreateConstant { .. } = cmd {
            res_constants.insert(res.unwrap().clone());
        }

        true
    };

    
    proc.step_cmd(&mut func_track);

    // ========== Step through Cmds ========== 
    let mut func = |proc: &mut IRProcedure, c_idx: &mut usize| {
        let idx = *c_idx;        

        let IRProcedure {id, main } = proc;
        let cmd = main.get_mut(idx).unwrap();
        let deps = ir_to_dep(cmd);
        let res = ir_to_res(cmd);
        let mut replace_var: Option<(String, String)> = None;

        if let Some(result) = res {
            for &dep in deps.iter() {
                // current command is a +=, /=, -=, etc.
                if deps.contains(&result) { break; }

                // don't include constants (these locations don't take up any sort of memory in the slightest)
                if res_constants.contains(result) { break; }

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

                /*
                check whether cmd is not a Concat operation
                [id] = concat([id], [id_one]) is not allowed.
                Due to the way that matrix tracker works (it works recursively), it's easier if we can gaurantee seperate inputs id to the result id
                */
                if let IRCmds::Concat { .. } = cmd { break; }
                
                // We require these 3 commands to have unique result + input IDs as well.
                // If we allow same ids, then we can get a result like this: by = sum(by, dim=-1)
                // If you look at an access expression of something like this: 
                // M (id: by, access: #x) = Sum (Vec/X: 128, Reduce/Y: 2)  (M (id: by, access: ((#y << 7) + #x)))
                // by is being accessed + written at the same time (with different access expressions), which can cause a conflict.
                // This does hurt the memory optimizations, but we optimize memory within the kernel level as well
                if let IRCmds::DotProduct { .. } = cmd { break; }
                if let IRCmds::Sum { .. } = cmd { break; }
                if let IRCmds::Contigious { .. } = cmd { break; }

                // if so, set replace var
                replace_var = Some( (result.clone(), dep.clone()) );
                break;
            }
        }

        // =================== Replace ================
        if let Some( (a, b) ) = replace_var {
            // replace result
            replace_res_cmd(cmd, b.clone());

            // replace all references from a to b
            // re-iterating O(n^2) <-- could be faster since we already have the locations
            replace_ref(proc, &a, b.clone());

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
    
    // update metadata
}