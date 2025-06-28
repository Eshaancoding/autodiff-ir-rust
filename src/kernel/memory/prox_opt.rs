// Proximity optimization
// attempts to move statements closest to its definition.

use std::collections::HashMap;

use crate::kernel_decl::{KernelProcedure, Kernels};

pub fn prox_opt (proc: &mut KernelProcedure) {
    let mut f = |proc: &mut KernelProcedure, c_idx: &mut usize| {
        let cmd_idx = *c_idx;

        // ============ find latest definitions locations ============
        let mut res_loc: HashMap<String, usize> = HashMap::new();
        let mut dep_loc: HashMap<String, usize> = HashMap::new(); 
        for i in 0..cmd_idx {
            let i_cmd = proc.get(i).unwrap();
            let res = i_cmd.get_res();
            let deps = i_cmd.get_dep_id();

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
        for dep in cmd.get_dep_id() {
            if let Some(dep_idx) = res_loc.get(dep) {
                if res_location.is_none_or(|f| (*dep_idx+1) > f) {
                    res_location = Some(*dep_idx + 1);
                }
            }
        }
        
        // result --> deps location
        if let Some(result) = cmd.get_res() {
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
        if let Kernels::While { .. } = cmd { res_location = None; }
        if let Kernels::If { .. } = cmd { res_location = None; }

        // =================== successful res location -> move ================
        if let Some(loc) = res_location {
            let item = proc.remove(cmd_idx);
            proc.insert(loc, item);
        }

        true
    };

    proc.step_cmd(&mut f);
}