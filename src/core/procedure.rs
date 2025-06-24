use crate::{IRCmds, IRProcedure};
use std::slice::{Iter, IterMut};

impl IRProcedure {
    pub fn new (id: String) -> IRProcedure {
        IRProcedure { main: vec![], id }
    }

    pub fn push (&mut self, cmd: IRCmds) {
        self.main.push(cmd);
    }

    // applies function to main procedure and any nested procedures.
    pub fn apply<T> (&mut self, f: &mut T)
        where T: FnMut(&mut IRProcedure)
    {
        f(self);

        for cmd in self.iter_mut() {
            if let IRCmds::If { conditions, else_proc } = cmd {
                for (_, i_proc) in conditions {
                    i_proc.apply(f);
                }
                if let Some(e_proc) = else_proc {
                    e_proc.apply(f);
                }
            }
            else if let IRCmds::While { block, .. } = cmd {
                block.apply(f);
            }
        }
    }

    pub fn step_cmd<T> (&mut self, f: &mut T)
        where T: FnMut(&mut IRProcedure, &mut usize) -> bool
        // block id, current cmd -> should continue loop
    {
        let mut cmd_idx = 0;
        
        loop {
            let should_continue = f(self, &mut cmd_idx);

            if let IRCmds::While { block, .. } = self.main.get_mut(cmd_idx).unwrap() {
                block.step_cmd(f);
            }
            else if let IRCmds::If { conditions, else_proc } = self.main.get_mut(cmd_idx).unwrap() {
                for (_, i_proc) in conditions {
                    i_proc.step_cmd(f)
                } 
                if let Some(e_proc) = else_proc {
                    e_proc.step_cmd(f)
                }
            }

            if should_continue { cmd_idx += 1; }

            if cmd_idx == self.main.len() {
                break;
            }
        }
    }

    // wrapper over Vec<IRCmds>
    pub fn iter (&self) -> Iter<'_, IRCmds> {
        self.main.iter()
    }

    pub fn iter_mut (&mut self) -> IterMut<'_, IRCmds> { 
        self.main.iter_mut()
    }

    pub fn get_mut (&mut self, idx: usize) -> Option<&mut IRCmds> {
        self.main.get_mut(idx) 
    }

    pub fn get (&self, idx: usize) -> Option<&IRCmds> {
        self.main.get(idx) 
    }

    pub fn insert (&mut self, idx: usize, cmd: IRCmds) {
        self.main.insert(idx, cmd);
    }

    pub fn len (&self) -> usize {
        self.main.len()
    }

    pub fn remove (&mut self, idx: usize) -> IRCmds {
        self.main.remove(idx)
    }
}

