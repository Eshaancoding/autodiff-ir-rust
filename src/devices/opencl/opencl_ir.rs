use indexmap::IndexMap;

use crate::{to_kernel::to_kernel, Device, ValueData, IRCmds};

pub struct OpenCL {
}

impl OpenCL {
    pub fn new () -> OpenCL {
        OpenCL {  }
    }
}

impl Device for OpenCL {
    fn execute (&mut self, cmds: IndexMap<String, Vec<IRCmds>>) {
        let _ = to_kernel(&cmds); // use to kernel method
    }

    fn get_tensor (&self, id: &String) -> ValueData {
        ValueData::none()
    }
}