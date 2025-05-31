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
        IRCmds::Sum { a, dim, res } => {
            println!("{} = sum({}, dim={})", res, a, dim);
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
        IRCmds::BR { block_id } => {
            println!("BR {}", block_id.cyan());
        },
        IRCmds::BRE { block_id, a } => {
            println!("if {} == 1 --> BR {}", a, block_id.cyan());
        },
        IRCmds::BRZ { block_id, a } => {
            println!("if {} == 0 --> BR {}", a, block_id.cyan());
        },
        IRCmds::EX => {
            println!("EX");
        }
    }
} 

/*
main:
    (0): a = mat(dim: [5, 3], contents: 0.030, 0.020, 0.010, 0.040, 0.050, 0.063, 0.010, 0.100, 0.070, 0.020, 0.010, 0.080, 0.050, 0.030, 0.060)
    (1): b = mat(dim: [3], contents: 0.030, 0.010, 0.032)
    (2): c = mat(dim: [3, 2], contents: 0.022, 0.016, 0.007, 0.093, 0.013, 0.090)
    (3): d = mat(dim: [2, 5], contents: 0.010, 0.020, 0.037, 0.040, 0.050, 0.063, 0.070, 0.080, 0.090, 0.013)
    (4): e = 0
    (5): f = 10
    (6): h = -1
    (7): i = h * f
    (8): m = 1
    (9): o = m.view(dim=[1, 1])
    (10): o = o.broadcast(dim=0, r=2)
    (11): o = o.broadcast(dim=1, r=3)
    (12): u = h.view(dim=[1, 1])
    (13): v = u.broadcast(dim=0, r=2)
    (14): v = v.broadcast(dim=1, r=3)
    (15): af = mat(dim: [2, 2], contents: 1.000, 1.000, 1.000, 1.000)
    (16): ax = 2
    (17): bm = d.T
    (18): bs = 0.1
    (19): bv = u.broadcast(dim=0, r=5)
    (20): bv = bv.broadcast(dim=1, r=3)
    (21): bx = bs.view(dim=[1, 1])
    (22): by = bx.broadcast(dim=0, r=5)
    (23): by = by.broadcast(dim=1, r=3)
    (24): h = h.broadcast(dim=0, r=3)
    (25): bs = bs.broadcast(dim=0, r=3)
    (26): u = u.broadcast(dim=0, r=3)
    (27): u = u.broadcast(dim=1, r=2)
    (28): bx = bx.broadcast(dim=0, r=3)
    (29): bx = bx.broadcast(dim=1, r=2)
    (30): BR g_while

g_while:
    (0): j = e + i
    (1): j = j < 0
    (2): if j == 0 --> BR g_while_end
    (3): 
    === Forward ===
    (4): x = dot(d, a)
    (5): y = b.view(dim=[1, 3])
    (6): y = y.broadcast(dim=0, r=2)
    (7): x += y
    (8): x *= v
    (9): x = x.exp()
    (10): ac = o + x
    (11): ad = o / ac
    (12): ae = dot(ad, c)
    (13): 
    === Backward ===
    (14): ag = c.T
    (15): ag = dot(af, ag)
    (16): ag *= v
    (17): ag *= o
    (18): ac = pow(ac, ax)
    (19): ag /= ac
    (20): ag *= x
    (21): ag *= v
    (22): bk = a.T
    (23): bl = dot(ag, bk)
    (24): bn = dot(bm, ag)
    (25): ag = sum(ag, dim=0)
    (26): bp = ag.view(dim=[3])
    (27): ad = ad.T
    (28): br = dot(ad, af)
    (29): 
    === Stepping ===
    (30): ca = by * bn
    (31): ca *= bv
    (32): a += ca
    (33): cg = bs * bp
    (34): cg *= h
    (35): b += cg
    (36): cq = bx * br
    (37): cq *= u
    (38): c += cq
    (39): e += m
    (40): BR g_while

g_while_end:
    (0): EX
*/