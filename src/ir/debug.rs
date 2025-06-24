use crate::{IRCmds, IRProcedure};
use colored::Colorize;
use std::fmt::{Display, Formatter};

pub fn print_while (f: &mut Formatter<'_>, cmd: &IRCmds, indent: usize) -> bool {
    if let IRCmds::While { conditional_var, block } = cmd {
        write!(f, "\n").expect("Can't write");
        for _ in 0..indent { write!(f, "    ").expect("Can't write"); }

        write!(f, "while ({} != 0) {{\n", conditional_var).expect("Can't write");
        print_proc(f, block, indent+1);

        for _ in 0..indent { write!(f, "    ").expect("Can't write"); }
        write!(f, "}}\n").expect("can't write");

        return true
    }
    false
}

pub fn print_if (f: &mut Formatter<'_>, cmd: &IRCmds, indent: usize) -> bool {
    if let IRCmds::If { conditions, else_proc } = cmd {

        write!(f, "\n").expect("Can't write");
        for _ in 0..indent { write!(f, "\n    ").expect("Can't write"); }

        for (idx, (condition, block)) in conditions.iter().enumerate() {
            if idx == 0 {
                write!(f, "if ({} == 1.0) {{\n", condition).expect("Can't write");
                print_proc(f, block, indent+1);

                for _ in 0..indent { write!(f, "    ").expect("Can't write"); }
                write!(f, "}}\n").expect("Can't write");
            } else {
                write!(f, "else if ({} == 1.0) {{", condition).expect("Can't write");
                print_proc(f, block, indent+1);

                for _ in 0..indent { write!(f, "    ").expect("Can't write"); }
                write!(f, "}}\n").expect("Can't write");
            }
        }
        if let Some(else_p) = else_proc {
            for _ in 0..indent { write!(f, "    ").expect("Can't write"); }
            write!(f, "else {{\n").expect("Can't write"); 
            print_proc(f, else_p, indent+1);
            
            for _ in 0..indent { write!(f, "    ").expect("Can't write"); }
            write!(f, "}}\n").expect("Can't write");
        } 
        
        return true
    }

    false
}

pub fn print_proc (f: &mut Formatter<'_>, proc: &IRProcedure, indent: usize) {
    for (i, cmd) in proc.iter().enumerate() {
        if !print_while(f, cmd, indent) && !print_if(f, cmd, indent) {
            for _ in 0..indent { write!(f, "    ").expect("Can't write"); }
            write!(f, "({}): {}\n", i, cmd).expect("Can't write"); 
        }
    }
}

impl Display for IRCmds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IRCmds::CreateMat {dim, id, contents} => {
                if *dim == vec![1] {
                    write!(f, "{} = {}", id, contents[0])
                } else {
                    if dim.iter().product::<usize>() < 16 {
                        let formatted: Vec<String> = contents.iter() 
                            .map(|n| format!("{:.3}", n))
                            .collect();
                        write!(f, "{} = mat(dim: {:?}, contents: {})", id, dim, formatted.join(", "))
                    } else {
                        write!(f, "{} = mat(dim: {:?}, contents: ...)", id, dim)
                    }
                }
            },
            IRCmds::CreateConstant { contents, id, dim } => {
                write!(f, "{} = {} (dim: {:?})", id, contents, dim)
            },
            IRCmds::ElwMultiply {a, b, res} => {
                write!(f, "{} = {} * {}", res, a, b)
            },
            IRCmds::ElwAdd {a, b, res} => {
                write!(f, "{} = {} + {}", res, a, b)
            },
            IRCmds::ElwAddEq { s, o } => {
                write!(f, "{} += {}", s, o)
            },
            IRCmds::ElwMultiplyEq { s, o } => {
                write!(f, "{} *= {}", s, o)
            },
            IRCmds::EqualZero { a, res } => {
                write!(f, "{} = {} == 0", res, a)
            },
            IRCmds::MoreZero { a, res } => {
                write!(f, "{} = {} > 0", res, a)
            },
            IRCmds::LessZero { a, res } => {
                write!(f, "{} = {} < 0", res, a)
            },
            IRCmds::DotProduct {a, b, res} => {
                write!(f, "{} = dot({}, {})", res, a, b)
            },
            IRCmds::View { a, target_dim, res } => {
                write!(f, "{} = {}.view(dim={:?})", res, a, target_dim)
            },
            IRCmds::Index { a, index, dim, res } => {
                write!(f, "{} = {}[ind={}, dim={}]", res, a, index, dim)
            },
            IRCmds::Concat { a, b, dim, res } => {
                write!(f, "{} = concat({}, {}, dim={})", res, a, b, dim)
            },
            IRCmds::Permute { a, p, res } => {
                if *p == vec![1, 0] {
                    write!(f, "{} = {}.T", res, a)
                } else {
                    write!(f, "{} = permute({}, {:?})", res, a, p)
                }
            },
            IRCmds::Contigious { a, res } => {
                write!(f, "{} = {}.contigious()", res, a)
            }
            IRCmds::Exp2 { a, res } => {
                write!(f, "{} = {}.exp2()", res, a)
            },
            IRCmds::Log2 { a, res } => {
                write!(f, "{} = {}.log2()", res, a)
            },
            IRCmds::Sin { a, res } => {
                write!(f, "{} = {}.sin()", res, a)
            },
            IRCmds::Recip { a, res } => {
                write!(f, "{} = 1/{}", res, a)
            },
            IRCmds::Sqrt { a, res } => {
                write!(f, "{} = sqrt({})", res, a)
            },
            IRCmds::Sum { a, res } => {
                write!(f, "{} = sum({}, dim=-1)", res, a)
            },
            IRCmds::Broadcast { a, dim, r, res } => {
                write!(f, "{} = {}.broadcast(dim={}, r={})", res, a, dim, r)
            },
            IRCmds::Heading { cmt } => {
                write!(f, "{}", format!("\n=== {} ===", cmt).purple().blue())
            },
            IRCmds::EX => {
                write!(f, "{}", "EXIT".red())
            },
            IRCmds::While { .. } => {
                print_while(f, self, 0);
                write!(f, "")
            },
            IRCmds::If { .. } => {
                print_if(f, self, 0);
                write!(f, "")
            }
        }
    } 
}

impl Display for IRProcedure {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        print_proc(f, self, 0);
        write!(f,"")
    }
}