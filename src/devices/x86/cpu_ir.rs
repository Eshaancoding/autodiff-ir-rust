// Using Tensor Rs library, we can create matrix and create operations
// Only single threaded operations

use indexmap::IndexMap;
use crate::{to_kernel::to_kernel, Device, IRBase, IRCmds, ValueData};
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
        let _ = to_kernel(self, &cmds);
    }

    fn get_tensor (&self, _: &String) -> ValueData {
        // not implemented yet
        ValueData::none()  
    }

    fn ir_callback (&self, irb: &mut IRBase) {
        // The efficient dot product in X86 implementations requires the weight matrix to be in column-major, not row-major. Furthermore, it requires the weight matrix to be contigious
        // Therefore, before each dot product implementation, we will add a transpose and contigious operation to the B weight matrix. 
        
        // find idx to change
        let mut permute_idxs: Vec<(String, usize, String)> = vec![];
        for (block_name, b_cmds) in irb.cmds.iter() {
            for (i, cmd) in b_cmds.iter().enumerate() {
                if let IRCmds::DotProduct { b, .. } = cmd {
                    permute_idxs.push((block_name.clone(), i, b.clone()));
                }
            }
        }

        // change
        let mut offset = 0;
        for (block_name, idx, b_id) in permute_idxs {
            let new_idx = offset + idx;            
            let new_id = irb.unique_id();
            let new_id_cont = irb.unique_id();

            let block = irb.cmds.get_mut(&block_name).unwrap();
            let dp_cmd = block.get_mut(new_idx).unwrap();

            // change dp
            if let IRCmds::DotProduct { b, .. } = dp_cmd {
                *b = new_id_cont.clone(); 
            } else {
                panic!("Not a dp cmd!");
            }

            // insert permute and contigious operations
            block.insert(new_idx, IRCmds::Permute { 
                a: b_id, 
                p: vec![1,0], 
                res: new_id.clone()
            });

            block.insert(new_idx + 1, IRCmds::Contigious { 
                a: new_id, 
                res: new_id_cont 
            });

            offset += 2;
        }
    }

    fn dot_prod_shape (&self, a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
        // as we transpose B (weight matrix), we have to switch what dim we are swapping
        vec![a.first().unwrap().clone(), b.first().unwrap().clone()] 
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

    fn ir_callback (&self, _: &mut IRBase) {
        // doesn't need to change anything in terms of IR callback
    }
}

