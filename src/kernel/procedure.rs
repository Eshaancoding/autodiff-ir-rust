use std::collections::HashMap;

use super::kernel_decl::{Kernels, KernelProcedure};

impl KernelProcedure {
    pub fn new () -> KernelProcedure {
        KernelProcedure { cmd_computeinstr: HashMap::new() } 
    }

    pub fn add_block (&mut self, block_name: &String, cmds: Vec<Kernels>) {
        self.cmd_computeinstr.insert(block_name.clone(), cmds);
    }

    pub fn block (&mut self, name: &String) -> &Vec<Kernels> {
        self.cmd_computeinstr.get(name).expect("Unable to get block")
    }
}