// Function that converts from IRCmds to a "expression string" 
// basically the print string without the result id embedded
// This is only used for repeat_opt, where if the same calculation is repeated, then it removes the repeated op.
// If there's no expression, then it returns None

use crate::IRCmds;

pub fn ir_to_expr (cmd: &IRCmds) -> Option<String> {
    match cmd {
        // don't optimize create matrix
        IRCmds::CreateConstant { contents, dim, .. } => {
            Some(format!("const(f: {:?}, dim: {:?})", contents, dim))
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
        IRCmds::Contigious { a, .. } => {
            Some(format!("{}.contigious()", a))
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
        IRCmds::Sum { a, .. } => {
            Some(format!("sum({}, dim=-1)", a))
        },
        IRCmds::Broadcast { a, dim, r, .. } => {
            Some(format!("{}.broadcast(dim={}, r={})", a, dim, r))
        },
        _ => { None }
    }
}