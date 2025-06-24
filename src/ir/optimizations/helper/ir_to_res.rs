// Each IR Command basically has a "result".
// This function returns said result. It is the opposite of `ir_to_expr` in a way.

use crate::IRCmds;

pub fn ir_to_res (cmd: &IRCmds) -> Option<&String> {
    match cmd {
        IRCmds::CreateMat {id, ..} => { Some(id) },
        IRCmds::CreateConstant { id , ..} => { Some(id) },

        IRCmds::ElwMultiply {res, ..} => { Some(res) },
        IRCmds::ElwAdd {res, ..} => { Some(res)},

        IRCmds::ElwMultiplyEq { s, .. } => { Some(s) },
        IRCmds::ElwAddEq { s, .. } => { Some(s) },

        IRCmds::EqualZero { res, .. } => { Some(res) },
        IRCmds::MoreZero { res, .. } => { Some(res) },
        IRCmds::LessZero { res, .. } => { Some(res) },

        IRCmds::Sum { res, .. } => { Some(res) },
        IRCmds::DotProduct { res, ..} => { Some(res) },

        IRCmds::View { res, ..} => { Some(res) },
        IRCmds::Index { res, ..} => { Some(res) },
        IRCmds::Concat { res, ..} => { Some(res) },
        IRCmds::Permute { res, ..} => { Some(res) },
        IRCmds::Broadcast { res, ..} => { Some(res) },
        IRCmds::Contigious { res, .. } => { Some(res) },

        IRCmds::Exp2 { res, ..} => { Some(res) },
        IRCmds::Log2 { res, ..} => { Some(res) },
        IRCmds::Sin { res, ..} => { Some(res) },

        IRCmds::Recip { res, .. } => { Some(res) },
        IRCmds::Sqrt { res, .. } => { Some(res) },
        _ => { None }
    }
}