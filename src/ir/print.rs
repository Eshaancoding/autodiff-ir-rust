use crate::IRCmds;
use colored::Colorize;

pub fn print_ir (cmd: &IRCmds, current_heading: &mut String, idx: usize) {
    print!("    ({}): ", idx);
    match cmd {
        IRCmds::CreateMat {dim, id, contents} => {
            if *dim == vec![1] {
                println!("{} = {}", id, contents[0]);
            } else {
                if dim.iter().product::<usize>() < 16 {
                    let formatted: Vec<String> = contents.iter() 
                        .map(|n| format!("{:.3}", n))
                        .collect();
                    println!("{} = mat(dim: {:?}, contents: {})", id, dim, formatted.join(", "));
                } else {
                    println!("{} = mat(dim: {:?}, contents: ...)", id, dim);
                }
            }
        },
        IRCmds::CreateConstant { contents, id, dim } => {
            println!("{} = {} (dim: {:?})", id, contents, dim);
        },
        IRCmds::ElwMultiply {a, b, res} => {
            println!("{} = {} * {}", res, a, b);
        },
        IRCmds::ElwAdd {a, b, res} => {
            println!("{} = {} + {}", res, a, b);
        },
        IRCmds::ElwAddEq { s, o } => {
            println!("{} += {}", s, o)
        },
        IRCmds::ElwMultiplyEq { s, o } => {
            println!("{} *= {}", s, o)
        },
        IRCmds::EqualZero { a, res } => {
            println!("{} = {} == 0", res, a);
        },
        IRCmds::MoreZero { a, res } => {
            println!("{} = {} > 0", res, a);
        },
        IRCmds::LessZero { a, res } => {
            println!("{} = {} < 0", res, a);
        },
        IRCmds::DotProduct {a, b, res} => {
            println!("{} = dot({}, {})", res, a, b);
        },
        IRCmds::View { a, target_dim, res } => {
            println!("{} = {}.view(dim={:?})", res, a, target_dim);
        },
        IRCmds::Index { a, index, dim, res } => {
            println!("{} = {}[ind={}, dim={}]", res, a, index, dim);
        },
        IRCmds::Concat { a, b, dim, res } => {
            println!("{} = concat({}, {}, dim={})", res, a, b, dim);
        },
        IRCmds::Permute { a, p, res } => {
            if *p == vec![1, 0] {
                println!("{} = {}.T", res, a);
            } else {
                println!("{} = permute({}, {:?})", res, a, p);
            }
        },
        IRCmds::Contigious { a, res } => {
            println!("{} = {}.contigious()", res, a);
        }
        IRCmds::Exp2 { a, res } => {
            println!("{} = {}.exp2()", res, a);
        },
        IRCmds::Log2 { a, res } => {
            println!("{} = {}.log2()", res, a);
        },
        IRCmds::Sin { a, res } => {
            println!("{} = {}.sin()", res, a);
        },
        IRCmds::Recip { a, res } => {
            println!("{} = 1/{}", res, a);
        },
        IRCmds::Sqrt { a, res } => {
            println!("{} = sqrt({})", res, a);
        },
        IRCmds::Sum { a, res } => {
            println!("{} = sum({}, dim=-1)", res, a);
        },
        IRCmds::Broadcast { a, dim, r, res } => {
            println!("{} = {}.broadcast(dim={}, r={})", res, a, dim, r);
        },
        IRCmds::Heading { cmt } => {
            println!("{}", format!("\n    === {} ===", cmt).purple().blue());
            *current_heading = cmt.clone();
        },
        IRCmds::Subheading { h, cmt } => {
            if h.is_none() || *current_heading == h.clone().unwrap() {
                println!("{}", format!("\n    {}", cmt).underline().red());
            }
        },
        IRCmds::EX => {
            println!("{}", "EXIT".red());
        }
    }
} 