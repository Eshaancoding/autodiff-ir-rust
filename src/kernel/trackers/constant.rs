use std::collections::HashMap;

use crate::IRCmds;

pub struct ConstantTracker {
    pub vars: HashMap<String, f64>
}

impl ConstantTracker {
    pub fn new () -> ConstantTracker {
        ConstantTracker { vars: HashMap::new() }
    }

    pub fn step (&mut self, cmd: &IRCmds) { 
        // we ignore the dim
        if let IRCmds::CreateConstant { contents, id, .. } = cmd {
            self.vars.insert(id.clone(), *contents);
        }

        if let IRCmds::ElwMultiplyEq { s, o } = cmd {
            let o_var_option= self.vars.get(o).map(|s| s.clone());
            if let Some(o_var) = o_var_option {
                // constant *= constant
                if let Some(s_var) = self.vars.get_mut(s) {
                    *s_var *= o_var;
                }
            }
            else {
                // constant *= tensor 
                // constant turns into a tensor after this cmd
                if self.vars.contains_key(s) {
                    self.vars.remove(s);
                }
            }
        }

        if let IRCmds::ElwAddEq { s, o } = cmd {
            let o_var_option= self.vars.get(o).map(|s| s.clone());
            if let Some(o_var) = o_var_option {
                // constant *= constant
                if let Some(s_var) = self.vars.get_mut(s) {
                    *s_var *= o_var;
                }
            }
            else {
                // constant *= tensor 
                // constant turns into a tensor after this cmd
                if self.vars.contains_key(s) {
                    self.vars.remove(s);
                }
            }
        }

        // any other way or method?
    }

    pub fn get_f64 (&self, id: &String) -> Option<f64> {
        match self.vars.get(id) {
            None => None,
            Some(v) => Some(*v)
        }
    }
}