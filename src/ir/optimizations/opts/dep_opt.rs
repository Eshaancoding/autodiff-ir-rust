// Removes variables that is not used at all

use std::collections::HashMap;
use indexmap::IndexMap;
use crate::{core::ret_dep_list, IRCmds};

use super::helper::{ir_to_dep, ir_to_res};

pub fn dep_opt (cmds: &mut IndexMap<String, Vec<IRCmds>>) {
    
    let dep_list = ret_dep_list();

    loop {
        let mut var_tracker: HashMap<String, bool> = HashMap::new();
        let mut var_placement: HashMap<String, (String, IRCmds)> = HashMap::new();
        
        for (block_name, b_cmds) in cmds.iter() {
            for cmd in b_cmds {
                let deps = ir_to_dep(cmd.clone());
                let res = ir_to_res(cmd.clone());
                
                if let Some(result) = res {
                    if !dep_list.contains(&result) {
                        var_tracker.insert(result.clone(), false); // mark as not used.
                        var_placement.insert(result, (block_name.clone(), cmd.clone()));
                    }
                }

                for d in deps {
                    var_tracker
                        .entry(d.clone())
                        .and_modify(|v| *v = true);
                }
            } 
        }


        // delete variables
        let var_tracker: Vec<&String> = var_tracker
            .iter()
            .filter(|(_, &is_used)| !is_used)
            .map(|f| f.0)
            .collect();

        for var in var_tracker.iter() {
            let (block_name, cmd) = var_placement.get(*var).unwrap();
            let block = cmds.get_mut(block_name).unwrap();
            let idx = block.iter().position(|x| x == cmd).unwrap();
            block.remove(idx);
        }

        if var_tracker.len() == 0 {
            break;
        }
    }

}