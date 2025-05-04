use indexmap::IndexMap;

use crate::{to_kernel::to_kernel, Device, ValueData};

pub struct OpenCL {
    tensor_hmap: IndexMap<String, ValueData>
}

impl OpenCL {
    pub fn new () -> OpenCL {
        OpenCL { tensor_hmap: IndexMap::new() }
    }
}

impl Device for OpenCL {
    fn execute (&mut self, cmds: IndexMap<String, Vec<crate::IRCmds>>) {
        let _ = to_kernel(&cmds); // use to kernel method
    }

    fn get_tensor (&self, id: &String) -> ValueData {
        ValueData::none()
    }
}