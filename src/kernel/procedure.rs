use crate::kernel_decl::Kernels;
use std::slice::Iter;
use super::kernel_decl::{KernelProcedure};

impl KernelProcedure {
    pub fn new (kernels: Vec<Kernels>, id: String) -> KernelProcedure {
        KernelProcedure {
            kernels, id
        }
    }

    pub fn iter (&self) -> Iter<'_, Kernels> {
        self.kernels.iter()
    }
}