use std::collections::HashMap;
use indexmap::IndexMap;
use crate::IRCmds;

pub fn get_plus_eq_vars (cmds: &IndexMap<String, Vec<IRCmds>>) -> (IndexMap<String, bool>, HashMap<String, (String, IRCmds)>) {
    let mut plus_eq_vars: IndexMap<String, bool> = IndexMap::new();
    let mut plus_eq_location: HashMap<String, (String, IRCmds)> = HashMap::new();
    for (block_name, cmd_blocks) in cmds.iter() {
        for cmd in cmd_blocks.iter() {
            if let IRCmds::ElwAddEq { s, .. } = cmd.clone() {
                plus_eq_vars.insert(s.clone(), true);
                plus_eq_location.insert(s, (block_name.clone(), cmd.clone()));
            }
            if let IRCmds::ElwMultiplyEq { s, .. } = cmd.clone() {
                plus_eq_vars.insert(s.clone(), true);
                plus_eq_location.insert(s, (block_name.clone(), cmd.clone()));
            }
        }
    }

    (plus_eq_vars, plus_eq_location)
}