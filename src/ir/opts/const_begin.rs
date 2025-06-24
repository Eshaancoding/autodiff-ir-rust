use std::collections::HashMap;

use crate::{IRCmds, IRProcedure};

// puts all the constants at the beginning
pub fn const_begin (procedure: &mut IRProcedure) {
    let mut const_tracker: Vec<(String, usize, IRCmds)> = vec![];

    // Track constant
    procedure.step_cmd(&mut |proc, idx| {
        let cmd = proc.get(*idx).unwrap();
        if let IRCmds::CreateConstant { .. } = cmd {
            const_tracker.push((proc.id.clone(), *idx, cmd.clone()));
        }

        true
    });

    
    // Delete constants at their location
    let mut delete_counter: HashMap<String, usize> = HashMap::new();
    const_tracker.sort_by(|a, b| a.1.cmp(&b.1));
    procedure.apply(&mut |proc| {
        for (proc_id, idx, _) in const_tracker.iter() {
            if proc.id == *proc_id {
                let r = delete_counter
                    .entry(proc_id.clone()) 
                    .and_modify(|v| *v += 1)
                    .or_insert(0);               

                proc.remove(*idx - *r);
            }
        }
    });

    // insert constants at the beginning
    for (_, _, cmd) in const_tracker {
        procedure.main.insert(0, cmd.clone());
    }
}