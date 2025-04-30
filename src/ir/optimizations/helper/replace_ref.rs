// Basically replace each reference from one variable to the other
// example: a += b. We want b to be replaced to c; Result: a += c

use indexmap::IndexMap;

use crate::IRCmds;

// replace a --> b 
fn replace_ref_cmd (cmd: &mut IRCmds, a_replace: &String, b_replace: String) {
    match cmd {
        IRCmds::ElwMultiply { a, b, .. } => {
            if a == a_replace { *a = b_replace.clone(); }
            if b == a_replace { *b = b_replace; }
        },
        IRCmds::ElwAdd { a, b, .. } => {
            if a == a_replace { *a = b_replace.clone(); }
            if b == a_replace { *b = b_replace; }
        },
        IRCmds::ElwAddEq { o, s, .. } => {
            if o == a_replace { *o = b_replace.clone(); }
            if s == a_replace { *s = b_replace; }
        },
        IRCmds::ElwMultiplyEq { o, s, .. } => {
            if o == a_replace { *o = b_replace.clone(); }
            if s == a_replace { *s = b_replace; }
        },
        IRCmds::EqualZero { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::MoreZero { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::LessZero { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::DotProduct { a, b, .. } => {
            if a == a_replace { *a = b_replace.clone(); }
            if b == a_replace { *b = b_replace; }
        },
        IRCmds::Sum { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::View { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::Index { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::Concat { a, b, .. } => {
            if a == a_replace { *a = b_replace.clone(); }
            if b == a_replace { *b = b_replace; }
        },
        IRCmds::Permute { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::Broadcast { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::Exp2 { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::Log2 { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::Sin { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::Neg { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::Recip { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::Sqrt { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::BRE { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        IRCmds::BRZ { a, .. } => {
            if a == a_replace { *a = b_replace; }
        },
        _ => {  } // do nothing
    }
}

pub fn replace_ref (cmds: &mut IndexMap<String, Vec<IRCmds>>, a_replace: &String, b_replace: String) {
    for (_, b_cmds) in cmds {
        for cmd in b_cmds {
            replace_ref_cmd(cmd, a_replace, b_replace.clone());
        }
    }
}