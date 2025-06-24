use std::collections::HashMap;

use crate::{
    core::ret_dep_list, 
    ir::helper::{ir_to_dep, ir_to_res}, 
    IRProcedure
};


pub fn dep_opt (procedure: &mut IRProcedure, var_changed: &Vec<String>) -> usize {
    let mut deleted: usize = 0; 
    let dep_list = ret_dep_list();

    let mut var_tracker: HashMap<String, bool> = HashMap::new();
    let mut var_placement: HashMap<String, (String, usize)> = HashMap::new();
    
    let mut func = |proc: &mut IRProcedure| {
        for (idx, cmd) in proc.iter().enumerate() {
            let deps = ir_to_dep(cmd);
            let res = ir_to_res(cmd);
            
            if let Some(result) = res {
                if !dep_list.contains(result) && !var_changed.contains(result) {
                    var_tracker.insert(result.clone(), false); // mark as not used.
                    var_placement.insert(result.clone(), (proc.id.clone(), idx));
                }
            }

            for d in deps {
                var_tracker
                    .entry(d.clone())
                    .and_modify(|v| *v = true);
            }
        } 
    };

    procedure.apply(&mut func);

    let var_tracker: Vec<&String> = var_tracker
        .iter()
        .filter(|(_, &is_used)| !is_used)
        .map(|f| f.0)
        .collect();

    let mut var_tracker: Vec<&(String, usize)> = var_tracker
        .iter()
        .map(|&v| var_placement.get(v).unwrap() )
        .collect();

    // sort by location
    var_tracker.sort_by(|&a, &b| a.1.cmp(&b.1) );

    let mut delete_counter: HashMap<String, usize> = HashMap::new();

    let mut func = |proc: &mut IRProcedure| {
        for &(proc_id, loc) in var_tracker.iter() {
            if proc.id == *proc_id {
                let r = delete_counter
                    .entry(proc_id.clone()) 
                    .and_modify(|v| *v += 1)
                    .or_insert(0);

                proc.main.remove(*loc - *r);
                deleted += 1;
            }
        }
    };

    procedure.apply(&mut func);

    deleted
}