use std::collections::HashMap;

use crate::IRCmds;

pub struct ConstantTracker {
    pub vars: HashMap<String, (f64, String)>
}

impl ConstantTracker {
    pub fn new () -> ConstantTracker {
        ConstantTracker { vars: HashMap::new() }
    }

    pub fn step (&mut self, cmd: &IRCmds) { 
        
    }

    pub fn get_f64 (&self, id: &String) -> Option<f64> {
        match self.vars.get(id) {
            None => None,
            Some(v) => Some(v.0)
        }
    }
}