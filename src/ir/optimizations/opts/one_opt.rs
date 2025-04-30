// simple optimization: if *= or multiplied by 1, then remove computation

use indexmap::IndexMap;

use crate::{core::ret_dep_list, ir::optimizations::helper::replace_ref, IRCmds};

// assumes that every command has an unique id (before mem_opt).
pub fn one_opt (cmds: &mut IndexMap<String, Vec<IRCmds>>) {
    // get dep list    
    let dep_list = ret_dep_list();

    let block_names: Vec<String> = cmds.iter().map(|(f, _)| f.clone()).collect();
    let mut block_name_idx = 0;
    let mut cmd_idx = 0;

    // get list of ones constants
    let mut one_csts: Vec<String> = vec![];
    for (_, b_cmds) in cmds.iter() {
        for d in b_cmds.iter() {
            if let IRCmds::CreateMat { contents, dim, id } = d {
                if *dim == vec![1] && contents[0] == 1.0 {
                    one_csts.push(id.clone());
                }
            }
            if let IRCmds::View { a, res, .. } = d {
                if one_csts.contains(&a) {
                    one_csts.push(res.clone());
                }
            }
            if let IRCmds::Broadcast { a, res, .. } = d {
                if one_csts.contains(&a) {
                    one_csts.push(res.clone());
                }
            }
        }
    }

    loop {
        // =================== Get current block ================
        let block_name = block_names.get(block_name_idx).unwrap();
        let block_list = cmds.get_mut(block_name).unwrap();
        let curr_cmd = block_list.get(cmd_idx).unwrap();

        let mut should_del= false;
        let mut ref_to_replace: Option<(String, String)> = None;
        if let IRCmds::ElwMultiply { a, b, res } = curr_cmd {
            if !dep_list.contains(res) {
                if one_csts.contains(a) {
                    should_del = true;
                    ref_to_replace = Some((res.clone(), b.clone()));
                }
                if one_csts.contains(b) {
                    should_del = true;
                    ref_to_replace = Some((res.clone(), a.clone()));
                }
            }
        }
        if let IRCmds::ElwMultiplyEq { o, ..} = curr_cmd {
            if one_csts.contains(o){
                should_del = true;
            }
        }

        // =================== Get next command ================
        let mut bl_len = block_list.len();
        if should_del {
            block_list.remove(cmd_idx);
            bl_len -= 1;
            
            if let Some((a, b)) = ref_to_replace {
                replace_ref(cmds, &a, b);
            }

        } else {
            cmd_idx += 1; 
        }

        if cmd_idx == bl_len {
            cmd_idx = 0;
            block_name_idx += 1;
        }
        if block_name_idx == block_names.len() {
            break;
        }
    }
}