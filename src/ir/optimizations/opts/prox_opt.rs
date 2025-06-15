// Proximity optimization
// attempts to move statements closest to its definition.

use std::collections::HashMap;
use indexmap::IndexMap;
use crate::IRCmds;

use super::helper::{ir_to_dep, ir_to_res};

pub fn prox_opt (cmds: &mut IndexMap<String, Vec<IRCmds>>) -> bool {
    
    let block_names: Vec<String> = cmds.iter().map(|(f, _)| f.clone()).collect();
    let mut block_name_idx = 0;
    let mut cmd_idx = 0;

    let mut perfect = true;

    loop {
        // =================== Get current block ================
        let block_name = block_names.get(block_name_idx).unwrap();
        let block_list = cmds.get_mut(block_name).unwrap();

        // ============ find latest definitions locations ============
        let mut res_loc: HashMap<String, usize> = HashMap::new();
        let mut dep_loc: HashMap<String, usize> = HashMap::new(); 
        for i in 0..cmd_idx {
            let i_cmd = block_list.get(i).unwrap();
            let res = ir_to_res(i_cmd.clone());
            let deps = ir_to_dep(i_cmd.clone());

            if let Some(result) = res {
                res_loc
                    .entry(result) 
                    .and_modify(|f| *f = i)
                    .or_insert(i);
            }
            for d in deps {
                dep_loc
                    .entry(d) 
                    .and_modify(|f| *f = i)
                    .or_insert(i);
            }
        } 

        // =================== Find potential position to move ================
        let mut res_location: Option<usize> = None;
        let cmd = block_list.get_mut(cmd_idx).unwrap();
        
        // deps --> definition location 
        for dep in ir_to_dep(cmd.clone()) {
            if let Some(dep_idx) = res_loc.get(&dep) {
                if res_location.is_none_or(|f| (*dep_idx+1) > f) {
                    res_location = Some(*dep_idx + 1);
                }
            }
        }
        
        // result --> deps location
        if let Some(result) = ir_to_res(cmd.clone()) {
            if let Some(i) = dep_loc.get(&result) {
                if res_location.is_some_and(|f| i+1 > f) {
                    res_location = Some(i + 1);
                }
            }
        }

        // we not swapping anything if idx are the exact same
        if res_location.is_some_and(|f| cmd_idx == f ) {
            res_location = None; 
        }

        // don't swap control statements. These commands are very location specific
        if let IRCmds::While { .. } = cmd { res_location = None; }
        if let IRCmds::If { .. } = cmd { res_location = None; }

        // =================== successful res location -> move ================
        if let Some(loc) = res_location {
            let item = block_list.remove(cmd_idx);
            block_list.insert(loc, item);
            perfect = true;
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

    perfect
}