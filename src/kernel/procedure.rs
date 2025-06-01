use std::collections::HashMap;

use super::kernel_decl::{ComputeInstr, Procedure};

impl Procedure {
    pub fn new () -> Procedure {
        Procedure { cmd_computeinstr: HashMap::new() } 
    }

    pub fn add_block (&mut self, block_name: &String, cmds: Vec<ComputeInstr>) {
        self.cmd_computeinstr.insert(block_name.clone(), cmds);
    }

    pub fn block (&mut self, name: &String) -> &Vec<ComputeInstr> {
        self.cmd_computeinstr.get(name).expect("Unable to get block")
    }
}