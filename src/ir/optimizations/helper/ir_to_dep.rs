// Each IR Command basically has dependencies
// This function returns said result. It is the direct opposite of `ir_to_res`

use crate::IRCmds;

pub fn ir_to_dep (cmd: IRCmds) -> Vec<String> {
    match cmd {
        IRCmds::ElwMultiply {a, b, ..} => vec![a, b],
        IRCmds::ElwAdd {a, b, ..} => vec![a, b],
        
        IRCmds::ElwMultiplyEq { s, o, .. } => vec![s, o],
        IRCmds::ElwAddEq { s, o, .. } => vec![s, o],

        IRCmds::EqualZero { a, .. } => vec![a],
        IRCmds::MoreZero { a, .. } => vec![a],
        IRCmds::LessZero { a, .. } => vec![a],

        IRCmds::Sum { a, .. } => vec![a],

        IRCmds::DotProduct { a, b, .. } => vec![a, b],

        IRCmds::View { a, .. } => vec![a],
        IRCmds::Index { a, .. } => vec![a],
        IRCmds::Concat { a, b, .. } => vec![a, b],
        IRCmds::Permute { a, .. } => vec![a],
        IRCmds::Broadcast { a, .. } => vec![a],
        IRCmds::Contigious { a, .. } => vec![a],

        IRCmds::Exp2 { a, .. } => vec![a],
        IRCmds::Log2 { a, .. } => vec![a],
        IRCmds::Sin { a, .. } => vec![a],
        IRCmds::Recip { a, .. } => vec![a],
        IRCmds::Sqrt { a, .. } => vec![a],

        IRCmds::BRE { a, .. } => { vec![a] },
        IRCmds::BRZ { a, .. } => { vec![a] },

        _ => { vec![] }
    }
}