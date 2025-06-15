// Proximity reverse optimization
// attempts to move definitions closest to its most recent moves

use std::collections::HashMap;
use indexmap::IndexMap;
use crate::IRCmds;

use super::helper::{ir_to_dep, ir_to_res};

pub fn prox_rev_opt (cmds: &mut IndexMap<String, Vec<IRCmds>>) -> bool {
    let block_names: Vec<String> = cmds.iter().map(|(f, _)| f.clone()).collect();
    let mut block_name_idx = 0;
    let mut cmd_idx = 0;
    let mut perfect = false;
    let mut swap_tracker: HashMap<(usize, usize), usize> = HashMap::new();

    loop {
        let mut did_change = false;        

        // =================== Get current block ================
        let block_name = block_names.get(block_name_idx).unwrap();
        
        // =================== Check for chain ================
        // However, there is a seperate optimization for chains alone, we are doing just a prelim check.
        let mut chain: Vec<(usize, usize)> = vec![];
        let mut init_ch: Option<(String, usize)> = None;
        for (idx, cmd) in cmds.get(block_name).unwrap().iter().enumerate() {
            if let Some(result) = ir_to_res(cmd.clone()) {
                match &init_ch {
                    None => { 
                        init_ch = Some((result, idx));
                    },
                    Some((start_res, start_idx)) => {
                        if result != *start_res {
                            if *start_idx != idx-1 { chain.push((*start_idx, idx-1)); }
                            init_ch = Some((result, idx));
                        }
                    }
                } 
            }
        } 
        
        // =================== Check if we can get swap ================
        let block_list = cmds.get_mut(block_name).unwrap();
        let cmd = block_list.get(cmd_idx).unwrap();
        if let Some(dep) = ir_to_res(cmd.clone()) {
            let current_deps = ir_to_dep(cmd.clone());

            // ============ find latest definitions locations ============
            let mut earliest_pos: Option<usize> = None;
            for i in (cmd_idx+1)..block_list.len() {
                let pot_cmd = block_list.get(i).unwrap();
                let pot_ref = ir_to_dep(pot_cmd.clone());
                let pot_res = ir_to_res(pot_cmd.clone());

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
                swap_tracker.get(&(cmd_idx, loc-1)).is_some_and(|v| *v == 2)
            }) {
                earliest_pos = None; 
            }

            // don't swap Control statements. These commands are very location specific
            if let IRCmds::While { .. } = cmd { earliest_pos = None; }
            if let IRCmds::If { .. } = cmd { earliest_pos = None; }

            // success res location -> swap
            if let Some(loc) = earliest_pos {
                let item = block_list.remove(cmd_idx);
                block_list.insert(loc-1, item);
                did_change = true;
                
                perfect = false;
                
                // insert into swap tracker
                swap_tracker
                    .entry((cmd_idx, loc-1))
                    .and_modify(|f| *f += 1)
                    .or_insert(1);
            }
        } 
        // =================== Get next command ================
        if !did_change {
            cmd_idx += 1; 
            if cmd_idx == cmds.get(block_name).unwrap().len() {
                cmd_idx = 0;
                block_name_idx += 1;
                swap_tracker.clear();
            }
            if block_name_idx == block_names.len() {
                break;
            }
        }
    }

    perfect
}