use crate::{to_kernel::to_kernel, Device, IRProcedure, ValueData};

pub struct OpenCL {
}

impl OpenCL {
    pub fn new () -> OpenCL {
        OpenCL { 
            
        }
    }
}

impl Device for OpenCL {
    fn execute (&mut self, cmds: &IRProcedure) {
        let _ = to_kernel(self, cmds);
    }

    fn get_tensor (&self, _: &String) -> ValueData {
        // not implemented yet
        ValueData::none()  
    }

    fn ir_callback (&self, cmds: &mut crate::IRBase) {}
}