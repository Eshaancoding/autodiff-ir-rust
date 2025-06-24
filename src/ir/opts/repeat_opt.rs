use std::collections::HashMap;

use crate::{ir::helper::{ir_to_expr, ir_to_res, replace_ref_cmd}, IRProcedure};

pub fn repeat_opt (procedure: &mut IRProcedure, var_changed: &Vec<String>) -> usize {
    
    let mut total_changed: usize = 0;
    
    let mut func = |proc: &mut IRProcedure| {
        let mut tracker: HashMap<String, Vec<usize>> = HashMap::new();  
        
        // get the exprs that are the exact same and track location
        for (idx, cmd) in proc.iter().enumerate() {
            let expr = ir_to_expr(cmd).unwrap_or("".to_string());
            if expr.len() == 0 { continue; }
            
            tracker.entry(expr)
                .and_modify(|v| v.push(idx))
                .or_insert(vec![idx]);
        }

        // filter variables that are only two or more locations
        let tracker: Vec<(&String, &Vec<usize>)> = tracker.iter().filter(|&(_, v)| v.len() > 1).collect();
        
        // track what variables to replace with and what idxs to delete
        let mut replace_to_res: Vec<(String, String)> = vec![];
        let mut delete_idxs: Vec<usize> = vec![];
        
        for (_, locs) in tracker {
            let mut first_res: Option<&String> = None;
            for i in 0..locs.len() {
                let idx = locs[i];
                let current_res = ir_to_res(proc.get(idx).unwrap()).unwrap();
                
                if i > 0 {
                    if var_changed.contains(&current_res) {
                        continue;  // don't add to delete idxs or replace
                    }
                    replace_to_res.push((current_res.clone(), first_res.unwrap().clone()));
                    delete_idxs.push(idx);
                } else {
                    first_res = Some(current_res);

                    if var_changed.contains(&current_res) { 
                        break; 
                        // don't change any of the variables that changes throughout the program
                        // go on to the next var to replace
                    }
                }
            }
        }

        // delete idxs
        delete_idxs.sort();
        for (i, idx) in delete_idxs.iter().enumerate() {
            proc.main.remove(idx - i);
            total_changed += 1;
        }

        // replace variables
        for cmd in proc.iter_mut() {
            for (to_search, to_replace) in replace_to_res.iter() {
                replace_ref_cmd(cmd, to_search, to_replace.clone());
            }
        }
    };

    procedure.apply(&mut func);

    total_changed
}