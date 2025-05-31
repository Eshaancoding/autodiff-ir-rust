// Function that converts from IRCmds to a "expression string" 
// basically the print string without the result id embedded
// If there's no expression, then it returns None

use crate::IRCmds;

pub fn ir_to_expr (cmd: &IRCmds) -> Option<String> {
    match cmd {
        IRCmds::CreateMat {dim, contents, ..} => {
            Some(format!("mat(dim: {:?}, contents: {:?})", dim, contents))
        },
        IRCmds::ElwMultiply {a, b, .. } => {
            Some(format!("{} * {}", a, b))
        },
        IRCmds::ElwAdd {a, b, .. } => {
            Some(format!("{} + {}", a, b))
        },
        IRCmds::ElwAddEq { s, o } => {
            Some(format!("{} += {}", s, o))
        },
        IRCmds::ElwMultiplyEq { s, o } => {
            Some(format!("{} *= {}", s, o))
        },
        IRCmds::EqualZero { a, ..  } => {
            Some(format!("{} == 0", a))
        },
        IRCmds::MoreZero { a, ..  } => {
            Some(format!("{} > 0", a))
        },
        IRCmds::LessZero { a, ..  } => {
            Some(format!("{} < 0", a))
        },
        IRCmds::DotProduct {a, b, .. } => {
            Some(format!("dot({}, {})", a, b))
        },
        IRCmds::View { a, target_dim, .. } => {
            Some(format!("{}.view(dim={:?})", a, target_dim))
        },
        IRCmds::Index { a, index, dim, .. } => {
            Some(format!("{}[ind={}, dim={}]", a, index, dim))
        },
        IRCmds::Concat { a, b, dim, .. } => {
            Some(format!("concat({}, {}, dim={})", a, b, dim))
        },
        IRCmds::Permute { a, p, .. } => {
            Some(format!("permute({}, {:?})", a, p))
        },
        IRCmds::Exp2 { a, .. } => {
            Some(format!("{}.exp()", a))
        },
        IRCmds::Log2 { a, .. } => {
            Some(format!("{}.ln()", a))
        },
        IRCmds::Sin { a, .. } => {
            Some(format!("{}.sin()", a))
        },
        IRCmds::Sqrt { a, ..} => {
            Some(format!("sqrt({a})"))
        }
        IRCmds::Recip { a, .. } => {
            Some(format!("1/{}", a))
        }
        IRCmds::Sum { a, dim, .. } => {
            Some(format!("sum({}, dim={})", a, dim))
        },
        IRCmds::Broadcast { a, dim, r, .. } => {
            Some(format!("{}.broadcast(dim={}, r={})", a, dim, r))
        },
        _ => { None }
    }
}