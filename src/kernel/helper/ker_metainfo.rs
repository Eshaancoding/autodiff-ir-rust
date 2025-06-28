use std::collections::HashMap;

use crate::kernel_decl::{Input, KernelProcedure, Kernels};

impl Input {
    pub fn get_id (&self) -> Vec<&String> {
        match self {
            Input::ConcatMatrix { id_one, id_two, .. } => {
                vec![id_one.get_id(), id_two.get_id()].concat()
            },
            Input::Mat { mat } => {
                vec![&mat.id]
            },
            Input::Constant { .. } => {
                vec![]
            }
        }
    }

    pub fn change_id (&mut self, match_id: &String, to_change: String) {
        match self {
            Input::ConcatMatrix { id_one, id_two, .. } => {
                id_one.change_id(match_id, to_change.clone()); 
                id_two.change_id(match_id, to_change); 
            },
            Input::Mat { mat } => {
                if mat.id == *match_id { mat.id = to_change; }
            }
            Input::Constant { .. } => {}
        }
    }

    pub fn get_mat_id (&self) -> Option<&String> {
        match self {
            Input::Mat { mat } => {
                Some(&mat.id)
            }, 
            // Ignore concatenation matrix at the result; not memory optimization that too much
            // (no clue HOW to optimize that in the first place.)
            _ => { None }
        }
    }
}


// Get dependencies and results
impl Kernels {
    // Get dependencies of the command
    pub fn get_dep (&self) -> Vec<&String> {
        match self {
            Kernels::Binary { a, b, .. } => {
                vec![a.get_id(), b.get_id()].concat()
            },
            Kernels::DotProd { a, b, .. } => {
                vec![a.get_id(), b.get_id()].concat()
            },
            Kernels::Unary { a, .. }  => { 
                a.get_id()
            },
            Kernels::Reduce { a, .. }  => { 
                a.get_id()
            },
            Kernels::Movement { a, .. }  => { 
                a.get_id()
            },
            Kernels::While { conditional_var, .. } => {
                vec![conditional_var]
            },
            Kernels::If { conditions, .. } => {
                let mut ret_str = vec![];
                for (cond, _) in conditions.iter() {
                    ret_str.push(cond);
                }

                ret_str
            },
            // Kernel fusion operations are created only after memory optimization 
            _ => { vec![] }
        }
    }

    pub fn change_dep (&mut self, match_id: &String, to_change: String) {
        match self {
            Kernels::Binary { a, b, .. } => {
                a.change_id(match_id, to_change.clone());
                b.change_id(match_id, to_change);
            },
            Kernels::DotProd { a, b, .. } => {
                a.change_id(match_id, to_change.clone());
                b.change_id(match_id, to_change);
            },
            Kernels::Unary { a, .. } => {   
                a.change_id(match_id, to_change);
            },
            Kernels::Reduce { a, .. } => {   
                a.change_id(match_id, to_change);
            },
            Kernels::Movement { a, .. } => {   
                a.change_id(match_id, to_change);
            },
            Kernels::While { conditional_var, .. } => {   
                if *conditional_var == *match_id {
                    *conditional_var = to_change
                }
            },
            Kernels::If { conditions, .. } => {
                for (cond, _) in conditions.iter_mut() {
                    if *cond == *match_id {
                        *cond = match_id.clone();
                    }
                }
            },
            _ => {}
        }
    }

    // Get resultant of the command
    pub fn get_res (&self) -> Option<&String> {
        match self {
            Kernels::Alloc { id, .. } => Some(&id),
            Kernels::Binary { res, .. } => Some(&res.id),
            Kernels::DotProd { res, .. } => Some(&res.id),
            Kernels::Unary { res, .. } => Some(&res.id),
            Kernels::Reduce { res, .. } => Some(&res.id),
            Kernels::Movement { res, .. } => Some(&res.id),
            _ => { None } // kernel fusion operations are only created after memory opt
        }
    }

    // Changed resultant id of the command
    pub fn change_res (&mut self, b: String) {
        match self {
            Kernels::Alloc { id, .. } => { *id = b; }
            Kernels::Binary { res, .. } => { res.id = b; },
            Kernels::DotProd { res, .. } => { res.id = b; },
            Kernels::Unary { res, .. } => { res.id = b; },
            Kernels::Reduce { res, .. } => { res.id = b; },
            Kernels::Movement { res, .. } => { res.id = b; },
            _ => {  } // kernel fusion operations are only created after memory opt
        }       
    }
}

impl KernelProcedure {
    // Get total vars changed
    pub fn get_var_changed (&mut self) -> Vec<String> {
        let mut var_changed: Vec<String> = vec![];
        let mut counter: HashMap<String, usize> = HashMap::new();

        let mut func = |proc: &mut KernelProcedure| {
            for cmd in proc.iter() {
                if let Kernels::Alloc { .. } = cmd { continue; }
                if let Kernels::Dealloc { .. } = cmd { continue; }

                let r = cmd.get_res();
                if let Some(res) = r {
                    counter.entry(res.clone())
                        .and_modify(|v| *v += 1)
                        .or_insert(1);
                }
                
                // *= or += ops
                if let Kernels::Binary { a, b, res, .. } = cmd {
                    if a.get_mat_id().is_some_and(|f| *f == *res.id) || b.get_mat_id().is_some_and(|f| *f == *res.id) {
                        var_changed.push(res.id.clone());
                    }
                }
            }
        };

        self.apply(&mut func);

        let to_append: Vec<(&String, &usize)> = counter.iter().filter(|&(_, v)| *v > 1).collect();
        let to_append: Vec<String> = to_append.iter().map(|&(i, _)| i.clone()).collect();
        
        for i in to_append {
            if !var_changed.contains(&i) {
                var_changed.push(i);
            }
        }

        var_changed
    }

    pub fn replace_ref (&mut self, match_id: &String, to_change: String) {
        self.step_cmd(&mut |proc, idx| {
            proc.get_mut(*idx).unwrap().change_dep(match_id, to_change.clone());
            true
        });        
    }

    pub fn remove_alloc (&self) -> KernelProcedure {
        let mut new = self.clone();

        new.apply(&mut |proc| {
            proc.filter(&mut |kernel| {
                if let Kernels::Alloc { .. } = kernel {
                    false
                }
                else if let Kernels::Dealloc { .. } = kernel { 
                    false
                }
                else {
                    true
                }
            })
        });

        new 
    }
}