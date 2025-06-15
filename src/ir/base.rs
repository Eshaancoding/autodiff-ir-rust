use colored::Colorize;
use crate::{IRBase, IRProcedure};
use super::{print::print_ir, IRCmds};

// IRBase to handle IR appending
impl IRBase {
    pub fn new () -> IRBase {
        IRBase {
            id: 0,
            current_block: "main".to_string(),
            main_block: "main".to_string(),
            proc: IRProcedure::new(),
            temp_proc: None
        }
    }

    pub fn unique_id_idx (idx: u32) -> String {
        let alphabet = "abcdefghijklmnopqrstuvwxyz";
        let base = alphabet.len() as u32;
        let mut result = String::new();
        let mut idx = idx;

        while idx >= base {
            result.push(alphabet.chars().nth((idx % base) as usize).unwrap());
            idx /= base;
            idx -= 1; 
        }
        result.push(alphabet.chars().nth(idx as usize).unwrap());

        result.chars().rev().collect() 
    }

    pub fn unique_id (&mut self) -> String {
        let res = IRBase::unique_id_idx(self.id);
        self.id += 1;
        res
    }

    pub fn create_temp_proc (&mut self) {
        self.temp_proc = Some(IRProcedure::new());
    }

    pub fn return_temp_proc (&mut self) -> IRProcedure {
        let c = self.temp_proc.clone().expect("Unwrapping procedure but none");
        self.temp_proc = None;
        c 
    }

    pub fn add_cmd (&mut self, cmd: IRCmds) {
        if let Some(proc) = &mut self.temp_proc { 
            proc.push(cmd);
        } else {
            self.proc.push(cmd);
        }
    }

    pub fn print (&self) {
        let mut current_heading = "".to_string();
        for (key, arr) in self.proc.iter() {        
            println!("{}:", key.cyan());
            for (i, cmd) in arr.iter().enumerate() {
                print_ir(cmd, &mut current_heading, i);
            }
            println!();
        }
        println!();     
    }

    pub fn create_block (&mut self, id: String) {
        self.current_block = id;
    }

    pub fn main_block (&mut self) {
        self.current_block = self.main_block.clone();
    }

    pub fn set_main_block (&mut self, id: String) {
        self.main_block = id;
    }
}