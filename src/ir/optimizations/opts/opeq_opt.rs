use indexmap::IndexMap;
use crate::IRCmds;

pub fn opeq_opt (cmds: &mut IndexMap<String, Vec<IRCmds>>) {
    let block_names: Vec<String> = cmds.iter().map(|(f, _)| f.clone()).collect();
    let mut block_name_idx = 0;
    let mut cmd_idx = 0;

    loop {
        // =================== Get current block ================
        let block_name = block_names.get(block_name_idx).unwrap();
        let cmd = cmds.get_mut(block_name).unwrap().get_mut(cmd_idx).unwrap();

        // =================== Replace Elw Add ================
        let mut rp: Option<(String, String)> = None;
        if let IRCmds::ElwAdd { a, b, res } = cmd {
            if res == a      { rp = Some((res.clone(), b.clone())); } 
            else if res == b { rp = Some((res.clone(), a.clone())); }
        }
        if let Some((s, o)) = rp {
            *cmd = IRCmds::ElwAddEq { s, o };
        }

        // =================== Replace Elw Mult ================
        let mut rp: Option<(String, String)> = None;
        if let IRCmds::ElwMultiply { a, b, res } = cmd {
            if res == a      { rp = Some((res.clone(), b.clone())); } 
            else if res == b { rp = Some((res.clone(), a.clone())); }
        }
        if let Some((s, o)) = rp {
            *cmd = IRCmds::ElwMultiplyEq { s, o };
        }

        // =================== Get next command ================
        cmd_idx += 1;
        if cmd_idx == cmds.get(block_name).unwrap().len() {
            cmd_idx = 0;
            block_name_idx += 1;
        }
        if block_name_idx == block_names.len() {
            break;
        }
    }
}