// Using Tensor Rs library, we can create matrix and create operations
// Only single threaded operations

use indexmap::IndexMap;
use crate::{to_kernel::to_kernel, Device, IRCmds, ValueData};
use super::exec::exec;
use tensor_rs::tensor_impl::gen_tensor::GenTensor;

pub struct CPUNew {
     
}

impl CPUNew {
    pub fn new () -> CPUNew {
        CPUNew { 
            
        }
    }
}

impl Device for CPUNew {
    fn execute (&mut self, cmds: IndexMap<String, Vec<IRCmds>>) {
        let _ = to_kernel(&cmds);
    }

    fn get_tensor (&self, id: &String) -> ValueData {
        // not implemented yet
        ValueData::none()  
    }
}

// ==================== tensor rs ==================== 
pub struct CPU {

    hmap: IndexMap<String, ValueData>,
}

impl CPU  {
    pub fn new () -> CPU {
        CPU  {
            hmap: IndexMap::new(),
        }
    }

    pub fn better_execute (&mut self, cmds: IndexMap<String, Vec<IRCmds>>) {
    }
}

impl Device for CPU {
    fn execute (&mut self, cmds: IndexMap<String, Vec<IRCmds>>) {
        let mut hms: IndexMap<String, GenTensor<f64>> = IndexMap::new();
        
        let mut c_idx: usize = 0;
        let mut c_block: String = "main".to_string();
        let mut c = &cmds[&c_block][c_idx];

        // simulate how a computer chip would execute EX and BRE instructions
        
        while *c != IRCmds::EX {
            // handle BR, BRE, or EX
            if let IRCmds::BR { block_id } = c {
                c_block = block_id.clone();
                c_idx = 0; 
            }
            else if let IRCmds::BRE { block_id, a } = c {
                if hms.get(a).unwrap().get_raw() == vec![1.0] {
                    c_block = block_id.clone(); 
                    c_idx = 0; 
                } 
                else {
                    c_idx += 1;
                }
            }
            else if let IRCmds::BRZ { block_id, a } = c {
                if hms.get(a).unwrap().get_raw() == vec![0.0] {
                    c_block = block_id.clone(); 
                    c_idx = 0; 
                } 
                else {
                    c_idx += 1;
                }
            }
            else {
                exec(c, &mut hms);            
                c_idx += 1;
            }
            c = &cmds[&c_block][c_idx];
        }
        
        // convert to Value Data
        for (key, value) in hms.iter() {
            self.hmap.insert(
                key.clone(),
                ValueData {
                    dim: value.size().clone(),
                    data: value.get_raw(),
                    id: key.clone(),
                    is_none: false
                }
            );
        }
    }

    fn get_tensor (&self, id:&String) -> ValueData {
        if let Some(x) = self.hmap.get(id) {
            return x.clone();
        } else {
            return ValueData::none()
        }
    }
}

