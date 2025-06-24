use crate::{IRCmds, IRProcedure};

pub fn opeq_opt (proc: &mut IRProcedure) {

    let mut f = |proc: &mut IRProcedure, idx: &mut usize| {
        let cmd = proc.get_mut(*idx).unwrap();

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

        true
    };

    proc.step_cmd(&mut f);
}