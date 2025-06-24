use crate::{IRBase, IRProcedure};
use super::IRCmds;

// IRBase to handle IR appending
impl IRBase {
    pub fn new () -> IRBase {
        let main_str = "main".to_string();

        IRBase { 
            id: 0,
            proc_id: 0,
            proc: IRProcedure::new(main_str),
            temp_proc: vec![]
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

    pub fn unique_id_proc (&mut self) -> String {
        let res = IRBase::unique_id_idx(self.proc_id);
        self.proc_id += 1;
        res
    }

    pub fn create_temp_proc (&mut self) {
        let u_id = self.unique_id_proc();
        self.temp_proc.push(IRProcedure::new(u_id));
    }

    pub fn return_temp_proc (&mut self) -> IRProcedure {
        let c = self.temp_proc.pop().expect("Unwrapping procedure but none");
        c 
    }

    pub fn add_cmd (&mut self, cmd: IRCmds) {
        if let Some(proc) = self.temp_proc.last_mut() { 
            proc.push(cmd);
        } else {
            self.proc.push(cmd);
        }
    }
}