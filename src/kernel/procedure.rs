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
}