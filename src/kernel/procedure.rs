use crate::kernel_decl::Kernels;
use std::slice::{Iter, IterMut};
use super::kernel_decl::{KernelProcedure};

// can we abstract from IR procedure?
impl KernelProcedure {
    pub fn new (kernels: Vec<Kernels>, id: String) -> KernelProcedure {
        KernelProcedure {
            kernels, id
        }
    }

    pub fn iter (&self) -> Iter<'_, Kernels> {
        self.kernels.iter()
    }

    pub fn insert (&mut self, idx: usize, cmd: Kernels) {
        self.kernels.insert(idx, cmd);
    }

    pub fn iter_mut (&mut self) -> IterMut<'_, Kernels> { 
        self.kernels.iter_mut()
    }

    pub fn remove (&mut self, idx: usize) -> Kernels {
        self.kernels.remove(idx)
    }

    pub fn get (&self, idx: usize) -> Option<&Kernels> {
        self.kernels.get(idx)
    }

    pub fn get_mut (&mut self, idx: usize) -> Option<&mut Kernels> {
        self.kernels.get_mut(idx)
    }

    pub fn len (&self) -> usize {
        self.kernels.len()
    }

    pub fn filter<P> (&mut self, predicate: &mut P)
    where 
        P: FnMut(&Kernels) -> bool
    {
        *self = KernelProcedure {
            kernels: self.kernels.iter().filter(|&p| predicate(p)).map(|f| f.clone()).collect(),
            id: self.id.clone()
        };
    }

    pub fn apply<T> (&mut self, f: &mut T)
        where T: FnMut(&mut KernelProcedure)
    {
        f(self);

        for cmd in self.iter_mut() {
            if let Kernels::If { conditions, else_proc } = cmd {
                for (_, i_proc) in conditions {
                    i_proc.apply(f);
                }
                if let Some(e_proc) = else_proc {
                    e_proc.apply(f);
                }
            }
            else if let Kernels::While { block, .. } = cmd {
                block.apply(f);
            }
        }
    }

    pub fn step_cmd<T> (&mut self, f: &mut T)
        where T: FnMut(&mut KernelProcedure, &mut usize) -> bool
        // block id, current cmd -> should continue loop
    {
        let mut cmd_idx = 0;
        
        loop {
            let should_continue = f(self, &mut cmd_idx);

            if let Kernels::While { block, .. } = self.kernels.get_mut(cmd_idx).unwrap() {
                block.step_cmd(f);
            }
            else if let Kernels::If { conditions, else_proc } = self.kernels.get_mut(cmd_idx).unwrap() {
                for (_, i_proc) in conditions {
                    i_proc.step_cmd(f)
                } 
                if let Some(e_proc) = else_proc {
                    e_proc.step_cmd(f)
                }
            }

            if should_continue { cmd_idx += 1; }

            if cmd_idx == self.kernels.len() {
                break;
            }
        }
    }
}