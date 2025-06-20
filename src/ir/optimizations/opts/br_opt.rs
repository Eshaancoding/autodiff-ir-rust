use std::collections::HashMap;
use indexmap::IndexMap;
use crate::IRCmds;

use super::helper::{get_plus_eq_vars, ir_to_dep::ir_to_dep, ir_to_res::ir_to_res};

// Attempts to move operations to the *latest* parent branch only on BR
/*
main:
    a = 1
    BR e_while

e_while:
    n = 1
    v = 3
    b = n * v
    a += 1
    ... break logic ...
    BR e_while 

notice that n, v, and b can be moved to the main branch. 
Therefore, for every iteration, we don't have to recompute that value.

main:
    a = 1
    n = 1
    v = 3
    b = n * v
    BR e_while

e_while:
    a += 1
    ... break logic ... 
    BR e_while

not that with BRE and BRZ branches, no optimization can be done. We let the user purely decide this. 
For example:

main:
    ... some logic ... 
    if [rare condition; x == 0] --> BR expensive_logic

expensive_logic:
    ... some expensive logic ... 
    BR end

end:
    EX

We don't want to move the expensive logic to main if we only satisfy the rare condition sometimes
*/

pub fn br_opt (cmds: &mut IndexMap<String, Vec<IRCmds>>) {
    // get parent branch 
    let mut br_to_parent: HashMap<String, (String, usize)> = HashMap::new();
    for (block_name, b_cmds) in cmds.clone() {
        for (i, cmd) in b_cmds.iter().enumerate() {
            // if let IRCmds::BR {block_id} = cmd.clone() {
            //     br_to_parent
            //         .entry(block_id)
            //         .or_insert((block_name.clone(), i));
            // }
        }
    }
    
    // track variables with plus equal
    let (plus_eq_vars, _) = get_plus_eq_vars(&cmds);

    // optimize
    loop {
        let mut max_var_to_move: Option<usize> = None;

        for (starting_block, (parent_block, idx_to_insert)) in br_to_parent.iter() {

            // track whether var changes within block
            let mut var_changes: IndexMap<String, bool> = plus_eq_vars.clone(); 
            let mut res_to_cmd: HashMap<String, IRCmds> = HashMap::new();

            // cmd
            for cmd in cmds.get(starting_block).unwrap() {
                let mut is_change = false;
                for dep in ir_to_dep(cmd.clone()) {
                    if var_changes.get(&dep).is_some_and(|f| *f) {
                        is_change = true;
                        break;
                    }
                }

                if let Some(res) = ir_to_res(cmd.clone()) {
                    if is_change {
                        var_changes
                            .entry(res.clone())
                            .and_modify(|v| *v = true)
                            .or_insert(true);
                    }
                    else {
                        var_changes
                            .entry(res.clone())
                            .or_insert(false);
                    }
                    res_to_cmd 
                        .entry(res)
                        .or_insert(cmd.clone());
                }
            }

            let var_changes: IndexMap<&String, &bool> = var_changes.iter().filter(|(_, &b)| !b).collect();
            let var_ch_len = var_changes.len();
            if let Some(max_var_move) = max_var_to_move {
                if var_ch_len > max_var_move {
                    max_var_to_move = Some(var_ch_len);
                } 
            } else {
                max_var_to_move = Some(var_ch_len);
            }

            let mut insert_counter: HashMap<&String, usize> = HashMap::new();

            for (&var_to_move, _) in var_changes.iter() {
                let cmd_from_var = res_to_cmd.get(var_to_move).unwrap().clone();
                let starting_block_cmds = cmds.get_mut(starting_block).unwrap();

                // remove the starting
                if let Some(index) = starting_block_cmds.iter().position(|x| *x == cmd_from_var) {
                    starting_block_cmds.remove(index);
                }

                // insert to cmd
                let z_usize: usize = 0;
                cmds.get_mut(parent_block).unwrap().insert(
                    idx_to_insert + insert_counter.get(parent_block).unwrap_or(&z_usize),
                    cmd_from_var 
                );

                insert_counter
                    .entry(parent_block)
                    .and_modify(|v| *v += 1)
                    .or_insert(1);
            }
        }

        if max_var_to_move.is_none_or(|x| x == 0) {
            break;
        }
    }    
}