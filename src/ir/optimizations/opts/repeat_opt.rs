use std::collections::HashMap;
use indexmap::IndexMap;
use crate::IRCmds;
use super::helper::{get_plus_eq_vars, ir_to_expr::*, ir_to_res::*, replace_ref::*};


// attempts to remove duplicate operations
/*
example:
a = -1
b = -1
c = 5
d = 4
...
x = a * c 
y = b * d

a and b are equal, so we can just use one memory location 
a = -1
c = 5
d = 4
...
x = a * c
y = a * d
*/

pub fn repeat_opt (cmds: &mut IndexMap<String, Vec<IRCmds>>) {
    loop {
        // have variables for a & b when found
        let mut a_result: Option<String> = None;
        let mut b_result = "".to_string();
        let mut b_block_name: String = "".to_string();
        let mut b_index: usize = 0;

        // track variables that have a plus equal in them.
        let (plus_eq_vars, _) = get_plus_eq_vars(&cmds);

        // convert all to expressions
        let mut expr_to_cmd: HashMap<String, String> = HashMap::new(); 
        
        for (block_name, cmds_block) in cmds.iter() {
            for (i, cmd) in cmds_block.iter().enumerate() {
                if let Some(expr) = ir_to_expr(cmd) {
                    if expr_to_cmd.contains_key(&expr) {
                        b_block_name = block_name.clone();
                        b_index = i;

                        b_result = ir_to_res(cmds[&b_block_name][b_index].clone()).unwrap();
                        a_result = Some(expr_to_cmd.get(&expr).unwrap().clone());

                        if plus_eq_vars.contains_key(&b_result) || plus_eq_vars.contains_key(&a_result.clone().unwrap()) {
                            // reset, in plus eq
                            a_result = None;
                            b_result = "".to_string();
                            b_block_name = "".to_string();
                            b_index = 0;
                            continue;
                        }
                    } else {
                        expr_to_cmd.insert(
                            expr.clone(),
                            ir_to_res(cmds[block_name][i].clone()).unwrap()
                        );
                    }
                }
                
                if a_result.is_some() { break; }
            }
            if a_result.is_some() { break; }
        }        
        if a_result.is_none() { break; }

        cmds.get_mut(&b_block_name).unwrap().remove(b_index);
        replace_ref(cmds, &b_result, a_result.unwrap());
    }

    
}