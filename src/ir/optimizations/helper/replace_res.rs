use crate::IRCmds;

pub fn replace_res_cmd (cmd: &mut IRCmds, replace_to: String) {
    match cmd {
        IRCmds::CreateMat { id, .. } => { *id = replace_to; },
        IRCmds::ElwMultiply { res, .. } => { *res = replace_to; },
        IRCmds::ElwAdd { res, .. } => { *res = replace_to; },

        IRCmds::ElwMultiplyEq { s, .. } => { *s = replace_to; },
        IRCmds::ElwAddEq { s, .. } => { *s = replace_to; },

        IRCmds::EqualZero { res, .. } => { *res = replace_to; },
        IRCmds::MoreZero { res, .. } => { *res = replace_to; },
        IRCmds::LessZero { res, .. } => { *res = replace_to; },

        IRCmds::Sum { res, .. } => { *res = replace_to; },
        IRCmds::DotProduct { res, .. } => { *res = replace_to; },

        IRCmds::View { res, .. } => { *res = replace_to; },
        IRCmds::Index { res, .. } => { *res = replace_to; },
        IRCmds::Concat { res, .. } => { *res = replace_to; },
        IRCmds::Permute { res, .. } => { *res = replace_to; },
        IRCmds::Broadcast { res, .. } => { *res = replace_to; },

        IRCmds::Exp2 { res, .. } => { *res = replace_to; },
        IRCmds::Log2 { res, .. } => { *res = replace_to; },
        IRCmds::Sin { res, .. } => { *res = replace_to; },

        IRCmds::Neg { res, .. } => { *res = replace_to; },
        IRCmds::Recip { res, .. } => { *res = replace_to; },
        IRCmds::Sqrt { res, .. } => { *res = replace_to; },

        _ => {}
    }
}
