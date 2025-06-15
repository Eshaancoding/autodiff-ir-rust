// Using Tensor Rs library, we can create matrix and create operations
// Only single threaded operations

use indexmap::IndexMap;
use crate::{to_kernel::to_kernel, Device, IRBase, IRCmds, IRProcedure, ValueData, IRB};
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
    fn execute (&mut self, cmds: IRProcedure) {
        // let _ = to_kernel(self, &cmds);
    }

    fn get_tensor (&self, _: &String) -> ValueData {
        // not implemented yet
        ValueData::none()  
    }

    fn ir_callback (&self, irb: &mut IRBase) {
        // The efficient dot product in X86 implementations requires the weight matrix to be in column-major, not row-major. Furthermore, it requires the weight matrix to be contigious
        // Therefore, before each dot product implementation, we will add a transpose and contigious operation to the B weight matrix. 
        
        let mut idx_offset = 0; // need to declare idx offset to satisfy borrow checker

        let mut f = |proc: &mut IRProcedure| { let mut permute_idxs: Vec<(usize, String)> = vec![];
            for (i, cmd ) in proc.iter().enumerate() {
                if let IRCmds::DotProduct { b, .. } = cmd {
                    permute_idxs.push((i, b.clone()));
                }
            }
            
            // change
            let mut offset = 0;
            for (idx, b_id) in permute_idxs {
                let new_idx = offset + idx;            
                let new_id = IRBase::unique_id_idx(irb.id + idx_offset);
                let new_id_cont = IRBase::unique_id_idx(irb.id + idx_offset + 1);
                idx_offset += 2;

                let dp_cmd = proc.get_mut(new_idx).unwrap();

                // change dp
                if let IRCmds::DotProduct { b, .. } = dp_cmd {
                    *b = new_id_cont.clone(); 
                } else {
                    panic!("Not a dp cmd!");
                }

                // insert permute and contigious operations
                proc.insert(new_idx, IRCmds::Permute { 
                    a: b_id, 
                    p: vec![1,0], 
                    res: new_id.clone()
                });

                proc.insert(new_idx + 1, IRCmds::Contigious { 
                    a: new_id, 
                    res: new_id_cont 
                });

                offset += 2;
            };
        };

        irb.proc.apply(&mut f);

        // then change irb idx 
        irb.id += idx_offset;
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

fn proc_exec (proc: &IRProcedure, hms: &mut IndexMap<String, GenTensor<f64>>) -> bool {
    let mut exit = false;    

    for cmd in proc.iter() {
        if let IRCmds::EX {} = cmd {
            return true;
        }
        else if let IRCmds::If { conditions, else_proc } = cmd {
            let mut run_cond = false;
            for (cond, c_proc) in conditions.iter() {
                if hms.get(cond).unwrap().get_data()[0] == 1.0 {
                    exit = proc_exec(c_proc, hms); // run whatever is inside condition
                    run_cond = true;               // set run condition
                    break;                         // don't eval any other conditions
                }
            }        

            if let Some(e_proc) = else_proc {
                if !run_cond { exit = proc_exec(e_proc, hms); } 
            }
        }
        else if let IRCmds::While { conditional_var, block } = cmd {
            while hms.get(conditional_var).unwrap().get_data()[0] != 0.0 {
                exit = proc_exec(block, hms);
                if exit { break; }
            }
        }
        else {
            exec(cmd, hms);            
        }

        if exit { return true; }
    }
    false
}

impl Device for CPU {
    fn execute (&mut self, cmds: IRProcedure) {
        let mut hms: IndexMap<String, GenTensor<f64>> = IndexMap::new();
        proc_exec(&cmds, &mut hms);
        
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

