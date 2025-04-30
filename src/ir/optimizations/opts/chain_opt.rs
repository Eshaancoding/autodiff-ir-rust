/*
a detection for commands in the "middle" of other commands, and move UP/DONE if possible.
allows for better intermediate fetching, so to speak.
in practice, this does little optimization, but it is good for peace-of-mind, we at least know that the compiler is optimizing for chains when it can; 
less dependent on the order or prox_opt and prox_rev_opt

    (11): ad = o / ac
    (12): ae = dot(ad, c)
    (13): ad = ad.T

    can't move (ad defined above/bolow and ad is used)

    (19): ag = c.T
    (20): c += ad
    (21): ag = dot(af, ag)   

    can move down

    (53): az = g * an
    (54): g *= y
    (55): az += bn
    (56): az += d
    (57): az += g   

    can't move up, but can move down (anyways stopped later by 57)

    (10): j += d
    (11): o = j.cos()
    (12): j = j.sin()   

    can't move (same as first example)
    
*/

use std::collections::HashSet;
use indexmap::IndexMap;
use crate::IRCmds;
use super::helper::{ir_to_dep, ir_to_res};

#[derive(PartialEq)]
enum Action {
    UP = 0,
    DOWN = 1,
    NONE = 2
}

pub fn chain_opt (cmds: &mut IndexMap<String, Vec<IRCmds>>) {
    let block_names: Vec<String> = cmds.iter().map(|(f, _)| f.clone()).collect();
    let mut block_name_idx = 0;
    let mut cmd_idx = 0;
    let block_name = block_names.get(block_name_idx).unwrap();
    let mut start = true;
    let mut go_up = false;
    
    let mut down_cmds: HashSet<usize> = HashSet::new();
    let mut up_cmds: HashSet<usize> = HashSet::new();

    loop {
        // =================== Get next command ================
        if !start { 
            if go_up {
                cmd_idx -= 1;
                go_up = false;
            } else {
                cmd_idx += 1;  
            }
        }
        start = false;
        if cmd_idx == cmds.get(block_name).unwrap().len() {
            cmd_idx = 0;
            block_name_idx += 1;
            down_cmds.clear();
            up_cmds.clear();
        }
        if block_name_idx == block_names.len() {
            break;
        }

        // =================== Get current block ================
        let block_name = block_names.get(block_name_idx).unwrap();
        let block_list = cmds.get_mut(block_name).unwrap();

        // ================ Chain detection code ================ 
        if cmd_idx == 0 { continue; }
        let next_cmd = match block_list.get(cmd_idx + 1) { Some(v) => v, None => { continue; } };
        let prev_cmd = match block_list.get(cmd_idx - 1) { Some(v) => v, None => { continue; } };
        let curr_cmd = match block_list.get(cmd_idx) { Some(v) => v, None => { continue; } };
        let next_cmd_res = match ir_to_res(next_cmd.clone()) { Some(v) => v, None => { continue; } };
        let prev_cmd_res = match ir_to_res(prev_cmd.clone()) { Some(v) => v, None => { continue; } };
        let curr_cmd_res = match ir_to_res(curr_cmd.clone()) { Some(v) => v, None => { continue; } };

        let prev_cmd_deps = ir_to_dep(prev_cmd.clone());
        let curr_cmd_deps = ir_to_dep(curr_cmd.clone());
        let next_cmd_deps = ir_to_dep(next_cmd.clone());

        // ================ Change order of cmd ================ 
        if next_cmd_res == prev_cmd_res && next_cmd_res != curr_cmd_res {
            let mut result_action = Action::UP;

            // check if we can go up
            if curr_cmd_deps.contains(&prev_cmd_res) { result_action = Action::DOWN; }
            if prev_cmd_deps.contains(&curr_cmd_res) { result_action = Action::DOWN; }
            if down_cmds.contains(&(cmd_idx-1)) { result_action = Action::DOWN; }

            if result_action == Action::DOWN {
                // check if we can go down
                if curr_cmd_deps.contains(&next_cmd_res) { result_action = Action::NONE; }
                if next_cmd_deps.contains(&curr_cmd_res) { result_action = Action::NONE; }
                if up_cmds.contains(&(cmd_idx+1)) { result_action = Action::NONE; }
            }

            if result_action == Action::UP {
                // change block list
                let item = block_list.remove(cmd_idx);
                block_list.insert(cmd_idx-1, item);

                // set alr set
                up_cmds.insert(cmd_idx);
            }
            else if result_action == Action::DOWN {
                // change block list
                let item = block_list.remove(cmd_idx);
                block_list.insert(cmd_idx+1, item);

                // set alr set
                down_cmds.insert(cmd_idx);
            } 

            // set the pointer of cmd to go up
            if result_action == Action::UP { go_up = true; }
        }

        
    }  
}

