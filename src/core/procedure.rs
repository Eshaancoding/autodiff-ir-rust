use crate::{IRCmds, IRProcedure};
use std::slice::{Iter, IterMut};

impl IRProcedure {
    pub fn new () -> IRProcedure {
        IRProcedure { main: vec![] }
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

    pub fn insert (&mut self, idx: usize, cmd: IRCmds) {
        self.main.insert(idx, cmd);
    }
}

