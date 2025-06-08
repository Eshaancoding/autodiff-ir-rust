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
        if let IRCmds::CreateMat { contents, dim, id } = cmd {
            if *dim == vec![1] {
                self.vars.insert(id.clone(), (contents[0], id.clone()));
            }
        }
        else if let IRCmds::View { a, res, .. } = cmd {
            if let Some(content) = self.vars.get(a) {
                self.vars.insert(res.clone(), content.clone());
            }
        }
        else if let IRCmds::Index { a, res , ..} = cmd {
            if let Some(content) = self.vars.get(a) {
                self.vars.insert(res.clone(), content.clone());
            }
        }
        else if let IRCmds::Permute { a, res, .. } = cmd {
            if let Some(content) = self.vars.get(a) {
                self.vars.insert(res.clone(), content.clone());
            }
        }
        else if let IRCmds::Broadcast { a, res, .. } = cmd {
            if let Some(content) = self.vars.get(a) {
                self.vars.insert(res.clone(), content.clone());
            }
        }
        else if let IRCmds::ElwAddEq { s, .. } = cmd {
            if self.vars.contains_key(s) {
                self.vars.retain(|_, x| 
                    x.1 != *s
                ); // delete source variables and any variables referencing source
            }
        }
        else if let IRCmds::ElwMultiplyEq { s, .. } = cmd {
            if self.vars.contains_key(s) {
                self.vars.retain(|_, x| 
                    x.1 != *s
                ); // delete source variables and any variables referencing source
            }
        }
    }

    pub fn get_f64 (&self, id: &String) -> Option<f64> {
        match self.vars.get(id) {
            None => None,
            Some(v) => Some(v.0)
        }
    }
}