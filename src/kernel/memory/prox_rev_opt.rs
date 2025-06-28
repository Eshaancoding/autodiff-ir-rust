use std::collections::HashMap;
use crate::kernel_decl::{KernelProcedure, Kernels};

pub fn prox_rev_opt (proc: &mut KernelProcedure) {
    let mut swap_tracker: HashMap<(String, usize, usize), usize> = HashMap::new();

    proc.step_cmd(&mut |proc: &mut KernelProcedure, c_idx: &mut usize| {
        let cmd_idx = *c_idx;
        let mut did_change = false;        

        // =================== Check for chain ================
        // However, there is a seperate optimization for chains alone, we are doing just a prelim check.
        let mut chain: Vec<(usize, usize)> = vec![];
        let mut init_ch: Option<(String, usize)> = None;
        for (idx, cmd) in proc.iter().enumerate() {
            if let Some(result) = cmd.get_res() {
                match &init_ch {
                    None => { 
                        init_ch = Some((result.clone(), idx));
                    },
                    Some((start_res, start_idx)) => {
                        if *result != *start_res {
                            if *start_idx != idx-1 { chain.push((*start_idx, idx-1)); }
                            init_ch = Some((result.clone(), idx));
                        }
                    }
                } 
            }
        } 
        
        // =================== Check if we can get swap ================
        let len_proc = proc.len();
        let cmd = proc.get(cmd_idx).unwrap();
        if let Some(dep) = cmd.get_res() {
            let current_deps = cmd.get_dep_id();

            // ============ find latest definitions locations ============
            let mut earliest_pos: Option<usize> = None;
            for i in (cmd_idx+1)..len_proc {
                let pot_cmd = proc.get(i).unwrap();
                let pot_ref = pot_cmd.get_dep_id();
                let pot_res = pot_cmd.get_res();

                // again, location specific statements 
                // can't move commands before/after while or if statements
                if let Kernels::While { .. } = pot_cmd { break; }
                if let Kernels::If { .. } = pot_cmd { break; }

                for r in pot_ref {
                    if r == dep {
                        earliest_pos = Some(i);
                        break; 
                    }
                }

                if let Some(res) = pot_res {
                    if current_deps.contains(&res) {
                        earliest_pos = Some(i);
                        break;
                    }
                }
                
                if earliest_pos.is_some() {
                    break;
                }
            }

            // if inside chain, then set it before chain is set.
            // prioritizes internal caching rather than memory length.
            if let Some(early_pos) = earliest_pos.as_mut() {
                let mut in_chain: Option<usize> = None;
                for (start, end) in chain.iter() {
                    if *early_pos > *start && *early_pos <= *end {
                        in_chain = Some(*start);
                        break;
                    } 
                }

                if let Some(mv_val) = in_chain {
                    *early_pos = mv_val;
                }
            }

            if earliest_pos.is_some_and(|f| f <= cmd_idx || f - cmd_idx == 1) {
                earliest_pos = None;
            } // nothing to swap.

            // if we swap 2x already, then we hit edge case. Don't switch
            if earliest_pos.is_some_and(|loc| {
                swap_tracker.get(&(proc.id.clone(), cmd_idx, loc-1)).is_some_and(|v| *v == 2)
            }) {
                earliest_pos = None; 
            }

            // don't swap Control statements. These commands are very location specific
            if let Kernels::While { .. } = cmd { earliest_pos = None; }
            if let Kernels::If { .. } = cmd { earliest_pos = None; }

            // success res location -> swap
            if let Some(loc) = earliest_pos {
                let item = proc.remove(cmd_idx);
                proc.insert(loc-1, item);
                did_change = true;
                
                // insert into swap tracker
                swap_tracker
                    .entry((proc.id.clone(), cmd_idx, loc-1))
                    .and_modify(|f| *f += 1)
                    .or_insert(1);
            }
        } 

        !did_change
    });
}