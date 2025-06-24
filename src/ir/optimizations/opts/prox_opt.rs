// Proximity optimization
// attempts to move statements closest to its definition.

use std::collections::HashMap;
use crate::{ir::optimizations::helper::{ir_to_dep, ir_to_res}, IRCmds, IRProcedure};

pub fn prox_opt (proc: &mut IRProcedure) {
    let mut f = |proc: &mut IRProcedure, c_idx: &mut usize| {
        let cmd_idx = *c_idx;

        // ============ find latest definitions locations ============
        let mut res_loc: HashMap<String, usize> = HashMap::new();
        let mut dep_loc: HashMap<String, usize> = HashMap::new(); 
        for i in 0..cmd_idx {
            let i_cmd = proc.get(i).unwrap();
            let res = ir_to_res(i_cmd);
            let deps = ir_to_dep(i_cmd);

            if let Some(result) = res {
                res_loc
                    .entry(result.clone()) 
                    .and_modify(|f| *f = i)
                    .or_insert(i);
            }
            for d in deps {
                dep_loc
                    .entry(d.clone()) 
                    .and_modify(|f| *f = i)
                    .or_insert(i);
            }
        } 

        // =================== Find potential position to move ================
        let mut res_location: Option<usize> = None;
        let cmd = proc.get_mut(cmd_idx).unwrap();
        
        // deps --> definition location 
        for dep in ir_to_dep(cmd) {
            if let Some(dep_idx) = res_loc.get(dep) {
                if res_location.is_none_or(|f| (*dep_idx+1) > f) {
                    res_location = Some(*dep_idx + 1);
                }
            }
        }
        
        // result --> deps location
        if let Some(result) = ir_to_res(cmd) {
            if let Some(i) = dep_loc.get(result) {
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
            let item = proc.main.remove(cmd_idx);
            proc.main.insert(loc, item);
        }

        true
    };

    proc.step_cmd(&mut f);
}