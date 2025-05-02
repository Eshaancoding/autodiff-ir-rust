use std::collections::HashMap;

use indexmap::IndexMap;
use crate::IRCmds;
use super::decl::{Kernels, Matrix};

pub fn to_kernel (cmds: &IndexMap<String, Vec<IRCmds>>) {
    let mut kernels: IndexMap<String, Vec<Kernels>> = IndexMap::new();
    let mut matrixes: HashMap<String, Matrix> = HashMap::new();
    let mut constants: HashMap<String, f64> = HashMap::new();
    let mut allocations: HashMap<String, usize> = HashMap::new(); // allocations to define with length of allocation provided.

    // 
    for (block_name, b_cmds) in cmds.iter() { 
        for cmd in b_cmds {
            
        }
    }
}