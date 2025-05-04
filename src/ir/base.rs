use indexmap::IndexMap;
use colored::Colorize;
use crate::IRBase;
use super::{print::print_ir, IRCmds};

impl IRBase {
    pub fn new () -> IRBase {
        IRBase {
            id: 0,
            current_block: "main".to_string(),
            main_block: "main".to_string(),
            cmds: IndexMap::new()
        }
    }

    pub fn unique_id (&mut self) -> String {
        let alphabet = "abcdefghijklmnopqrstuvwxyz";
        let base = alphabet.len() as u32;
        let mut idx = self.id;
        let mut result = String::new();

        while idx >= base {
            result.push(alphabet.chars().nth((idx % base) as usize).unwrap());
            idx /= base;
            idx -= 1; 
        }
        result.push(alphabet.chars().nth(idx as usize).unwrap());
        self.id += 1;

        result.chars().rev().collect() 
    }

    pub fn add_cmd (&mut self, cmd: IRCmds) {
        let v = self.cmds.entry(self.current_block.clone()).or_insert(vec![]);
        v.push(cmd);
    }

    pub fn print (&self) {
        let mut current_heading = "".to_string();
        for (key, arr) in self.cmds.iter() {        
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