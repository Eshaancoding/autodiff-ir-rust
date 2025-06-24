use std::collections::HashMap;

// tracks the changed variables
use crate::{ir::helper::ir_to_res, IRCmds, IRProcedure};

pub fn track_var_changed (procedure: &mut IRProcedure) -> Vec<String> {
    let mut var_changed: Vec<String> = vec![];
    let mut counter: HashMap<String, usize> = HashMap::new();

    let mut func = |proc: &mut IRProcedure| {
        for cmd in proc.iter() {
            let r = ir_to_res(cmd);
            if let Some(res) = r {
                counter.entry(res.clone())
                    .and_modify(|v| *v += 1)
                    .or_insert(1);
            }
            
            if let IRCmds::ElwMultiplyEq { s, .. } = cmd {
                var_changed.push(s.clone());
            }
            else if let IRCmds::ElwAddEq { s, .. } = cmd {
                var_changed.push(s.clone());
            }
        }
    };

    procedure.apply(&mut func);

    let to_append: Vec<(&String, &usize)> = counter.iter().filter(|&(_, v)| *v > 1).collect();
    let to_append: Vec<String> = to_append.iter().map(|&(i, _)| i.clone()).collect();
    
    for i in to_append {
        if !var_changed.contains(&i) {
            var_changed.push(i);
        }
    }

    var_changed
}