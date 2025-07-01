
use std::collections::HashMap;
use crate::{
    ir::helper::ir_to_res, trackers::ConstantTracker, Device, IRCmds
};
use super::ShapeTracker;

// Cmds that are only pertinent to data manipulation
#[derive(Clone, Debug)]
pub enum DataCmds {
    View { source_dim: Vec<usize>, sink_dim: Vec<usize> },
    Index { index: usize, dim: usize },
    Permute { p: Vec<usize> }, 
    Broadcast { dim: usize, r: usize }
}

#[derive(Clone, Debug)]
pub struct VarDependency {
    pub id: String,
    pub source_dims: Vec<usize>,
    pub data_cmds: Vec<DataCmds>
}

#[derive(Clone, Debug)]
pub struct VarSource {
    pub id: String,
    pub dim: Vec<usize>
}

#[derive(Clone, Debug)]
pub struct VarConcat {
    pub a: String,
    pub b: String,
    pub dim: usize,
    pub idx_end: usize,
}

#[derive(Clone, Debug)]
pub struct VarConcatDep {
    pub source: VarConcat,
    pub data_cmds: Vec<DataCmds>
}

#[derive(Clone)]
pub struct KernelTracker {
    // given sink variable, get source variable and the steps to reach to sink var
    // Vars and sources hashmap keys must always come from alloc tracker's id (except for concat vars)
    pub sources: HashMap<String, VarSource>,                // tracks source variables (no var dependency)
    pub vars: HashMap<String, VarDependency>,               // tracks the dependency of variables that are related to source (list of DataCmds)
    pub sources_concat: HashMap<String, VarConcat>,         // tracks concat variables 
    pub vars_concat: HashMap<String, VarConcatDep>,         // tracks variables that references concat variables

    pub shape_tracker: ShapeTracker,                        // tracks the shape of variables
    pub constant_tracker: ConstantTracker,                  // tracks constant tracker
}

// Different access types (depending on the kernel used) is needed
#[derive(PartialEq, Eq)]
pub enum AccessType {
    Global,         // Elementwise; uses Global IDX
    XY,             // Dot product/Reduce; restricted to matrix 2-dim; uses X and Y. (SEE src/matmul_cpu/512_matmul.cpp for example of x + y)
}

impl KernelTracker {
    pub fn new () -> KernelTracker {
        KernelTracker { 
            sources: HashMap::new(),
            vars: HashMap::new(), 
            sources_concat: HashMap::new(),
            vars_concat: HashMap::new(),            
            shape_tracker: ShapeTracker::new(),
            constant_tracker: ConstantTracker::new(),
        }
    }

    pub fn step (&mut self, device: &dyn Device, cmd: &IRCmds) {
        let mut prev_dim: Vec<usize> = vec![];
        if let IRCmds::View { a, .. } = cmd {
            prev_dim = self.shape_tracker.get_shape(&a).clone();
        }

        // needs to be sync with the Matrix Tracker
        self.shape_tracker.step(device, cmd);
        self.constant_tracker.step(cmd);

        // track the sources and the variables
        let mut dep_cmp: String = "".to_string();
        let mut res_cmp: String = "".to_string();        
        let mut data_clone: Option<(String, VarConcat)> = None;
        let mut data_cmd: Option<DataCmds> = None;
        
        if let IRCmds::View { a, res, .. } = cmd {
            let sink_dim = self.shape_tracker.get_shape(&res).clone();
            dep_cmp = a.clone();
            res_cmp = res.clone();
            data_cmd = Some(DataCmds::View { source_dim: prev_dim, sink_dim });
        }
        else if let IRCmds::Index { a, index, dim, res } = cmd {
            dep_cmp = a.clone();
            res_cmp = res.clone();
            data_cmd = Some(DataCmds::Index { index: index.clone(), dim: dim.clone() });
        }
        else if let IRCmds::Concat { a, b, dim, res } = cmd {
            assert!(res != a && res != b, "Res id can't be the same as a and b id at Concat");
            let idx_end = self.shape_tracker.get_shape(a)[*dim].clone();
            data_clone = Some((res.clone(), VarConcat {
                a: a.clone(), 
                b: b.clone(), 
                dim: *dim,
                idx_end,
            }));
        }
        else if let IRCmds::Permute { a, p, res } = cmd {
            dep_cmp = a.clone();
            res_cmp = res.clone();
            data_cmd = Some(DataCmds::Permute { p: p.clone() });
        }
        else if let IRCmds::Broadcast { a, dim, r, res } = cmd {
            dep_cmp = a.clone();
            res_cmp = res.clone();
            data_cmd = Some(DataCmds::Broadcast { dim: *dim, r: *r });
        } else {
            if let Some(id) = ir_to_res(cmd) {
                // only cmd satisfying this is CreateMat. If it's a constant, then skip
                if let Some(_) = self.constant_tracker.get_f32(&id) {
                    return     
                }

                // if we are redefining a source, then remove from self.vars (which tracks broadcasting, view, etc.)
                self.vars.remove_entry(id);

                let shape = self.shape_tracker.get_shape(&id).clone();
                self.sources.insert(
                    id.clone(), 
                    VarSource { 
                        id: id.clone(),
                        dim: shape
                    }
                );
            }
        }

        // Declaring a concat variable
        if let Some(d) = data_clone {
            self.sources_concat.insert(d.0, d.1);
        }
        else if let Some(cmds) = data_cmd {

            // =============== Variables =============== 
            /*
            (7): g = 1 <-- source var
            (8): o = g.view(dim=[1, 1]) <-- o is referencing source var o
             */
            if let Some(var_source) = self.sources.get(&dep_cmp) {
                self.vars.insert(
                    res_cmp.clone(),
                    VarDependency {
                        id: var_source.id.clone(),
                        source_dims: var_source.dim.clone(),
                        data_cmds: vec![cmds],
                    }
                );

                /*
                (12): l = 1.4426950408889634 
                (13): l = l.view(dim=[1, 1])
                 */
                if res_cmp == dep_cmp {
                    self.sources.remove_entry(&res_cmp);
                }
            }
            /*
            (27): bz = 0.1 <-- bz is source
            (29): ca = bz.view(dim=[1, 1]) <-- ca is in vars due to referencing bz
            (33): cb = ca.broadcast(dim=0, r=256) <-- cb is referencing ca in vars
             */
            else if let Some(dep) = self.vars.get_mut(&dep_cmp) { // referencing another variable that is in vars
                if res_cmp == dep_cmp {
                    /*
                    Covers edge case where both res and dep are the same
                    (14): l = l.broadcast(dim=0, r=2)
                    (15): l = l.broadcast(dim=1, r=128)           
                    */
                    dep.data_cmds.push(cmds);
                } else {
                    let mut dep = dep.clone();
                    dep.data_cmds.push(cmds);
                    self.vars.insert(
                        res_cmp.clone(),
                        dep
                    );
                }
            }
            // =============== Concat Variables =============== 
            /*
            (3): d = concat(b, c, dim=1) <-- d added to vars_concat
            (4): k = permute(d, [0, 1])  <-- k is is referencing d, which is part of vars_concat
            (5): l = k.contigious()
            */
            else if let Some(res) = self.sources_concat.get(&dep_cmp) {
                self.vars_concat.insert(
                    res_cmp.clone(),
                    VarConcatDep {
                        source: res.clone(),
                        data_cmds: vec![cmds],
                    }
                );

                if res_cmp == dep_cmp { self.sources_concat.remove_entry(&dep_cmp); }
            }
            /*
            (3): d = concat(b, c, dim=1) <-- d added to sources_concat
            (4): k = permute(d, [0, 1])  <-- k is referencing d. K is added to vars_concat
            (5): p = k.view([5, 2])      <-- p is referencing k, which is part of vars_concat
            */
            else if let Some(dep) = self.vars_concat.get_mut(&dep_cmp) {
                if res_cmp == dep_cmp {
                    dep.data_cmds.push(cmds);
                } else {
                    let mut dep = dep.clone();
                    dep.data_cmds.push(cmds);
                    self.vars_concat.insert(
                        res_cmp.clone(),
                        dep
                    );
                }
            }
            else {
                panic!("Unable to step through cmd in Matrix tracker at cmd: {}", cmd);
            }
        }
    }

    // wrapper over shape tracker
    pub fn get_shape (&self, id: &String) -> &Vec<usize> {
        self.shape_tracker.get_shape(id)
    }

    // wrapper over constant tracker
    pub fn get_constant (&self, id: &String) -> Option<f32> {
        self.constant_tracker.get_f32(id)
    }
}