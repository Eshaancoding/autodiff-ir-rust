/*
=== Matrix Tracker ===
Given an id, extract's the source matrix id, the expression needed to go to id.
Also, handles constants the concatenation matrixes + constants
This is how memory is handled within all backend kernels.
*/

use std::collections::HashMap;
use crate::{
    ir::optimizations::helper::ir_to_res, 
    IRCmds
};
use super::{AllocTracker, ShapeTracker};

// Cmds that are only pertinent to data manipulation
#[derive(Clone, Debug)]
pub enum DataCmds {
    View { source_dim: Vec<usize>, sink_dim: Vec<usize> },
    Index { index: usize, dim: usize } ,
    Concat,
    Permute { p: Vec<usize> }, 
    Broadcast { dim: usize, r: usize }
}

#[derive(Clone, Debug)]
pub struct VarDependency {
    pub alloc_id: String,
    pub source_dims: Vec<usize>,
    pub sink_dims: Vec<usize>,
    pub data_cmds: Vec<DataCmds>
}

#[derive(Clone, Debug)]
pub struct VarSource {
    pub alloc_id: String,
    pub dim: Vec<usize>
}

pub struct MatrixTracker<'a> {
    // given sink variable, get source variable and the steps to reach to sink var
    // Note that these IDs aren't correspondent to the HLIR id, but to the alloc ID.
    pub vars: HashMap<String, VarDependency>,  
    pub sources: HashMap<String, VarSource>,             // tracks source variables (no var dependency)
    pub shape_tracker: ShapeTracker,                     // tracks the shape of variables
    pub alloc_tracker: &'a AllocTracker<'a>
}

impl<'a> MatrixTracker<'a> {
    pub fn new (alloc_tracker: &'a AllocTracker) -> MatrixTracker<'a> {
        MatrixTracker { 
            vars: HashMap::new(), 
            sources: HashMap::new(),
            shape_tracker: ShapeTracker::new(),
            alloc_tracker
        }
    }

    pub fn step (&mut self, cmd: &IRCmds) {
        let mut prev_dim: Vec<usize> = vec![];
        if let IRCmds::View { a, .. } = cmd {
            prev_dim = self.shape_tracker.get_shape(&a).clone();
        }

        // needs to be sync with the Matrix Tracker
        // we declare seperate shape tracker at alloc tracker
        self.shape_tracker.step(cmd);

        // track the sources and the variables
        let mut a_cmp: String = "".to_string();
        let mut res_cmp: String = "".to_string();        
        let mut data_cmd: Option<DataCmds> = None;
        
        if let IRCmds::View { a, res, .. } = cmd {
            let sink_dim = self.shape_tracker.get_shape(&res).clone();
            a_cmp = a.clone();
            res_cmp = res.clone();
            data_cmd = Some(DataCmds::View { source_dim: prev_dim, sink_dim });
        }
        else if let IRCmds::Index { a, index, dim, res } = cmd {
            a_cmp = a.clone();
            res_cmp = res.clone();
            data_cmd = Some(DataCmds::Index { index: index.clone(), dim: dim.clone() });
        }
        else if let IRCmds::Concat { a, b, dim, res } = cmd {
            // concat is weird, not doing this yet
            todo!()
        }
        else if let IRCmds::Permute { a, p, res } = cmd {
            a_cmp = a.clone();
            res_cmp = res.clone();
            data_cmd = Some(DataCmds::Permute { p: p.clone() });
        }
        else if let IRCmds::Broadcast { a, dim, r, res } = cmd {
            a_cmp = a.clone();
            res_cmp = res.clone();
            data_cmd = Some(DataCmds::Broadcast { dim: *dim, r: *r });
        } else {
            if let Some(id) = ir_to_res(cmd.clone()) {
                // if we are redefining a source, then remove from self.vars (which tracks broadcasting, view, etc.)
                self.vars.remove_entry(&id);

                let shape = self.shape_tracker.get_shape(&id).clone();
                let alloc_id = self.alloc_tracker.get_alloc(&id).id.clone();
                self.sources.insert(
                    id, 
                    VarSource { 
                        alloc_id,
                        dim: shape
                    }
                );
            }
        }

        // All examples are from test::nn_test::tests::nn_time_test
        if let Some(cmd) = data_cmd {
            let sink_dims = self.shape_tracker.get_shape(&res_cmp).clone();

            /*
            (14): l = l.broadcast(dim=0, r=2)
            (15): l = l.broadcast(dim=1, r=128)           
             */
            if res_cmp == a_cmp && self.vars.contains_key(&res_cmp) { 
                let var_m = self.vars.get_mut(&res_cmp).unwrap();
                var_m.data_cmds.push(cmd);
                var_m.sink_dims = sink_dims;
            }
            /*
            (7): g = 1
            (8): o = g.view(dim=[1, 1])
             */
            else if self.sources.contains_key(&a_cmp) { // references sources
                let var_source = self.sources.get(&a_cmp).unwrap().clone();
                self.vars.insert(
                    res_cmp.clone(),
                    VarDependency {
                        alloc_id: var_source.alloc_id,
                        source_dims: var_source.dim,
                        data_cmds: vec![cmd],
                        sink_dims
                    }
                );

                /*
                (12): l = 1.4426950408889634
                (13): l = l.view(dim=[1, 1])
                 */
                if res_cmp == a_cmp {
                    self.sources.remove_entry(&res_cmp);
                }
            }
            /*
            (27): bz = 0.1
            (29): ca = bz.view(dim=[1, 1])
            (33): cb = ca.broadcast(dim=0, r=256)
             */
            else { // referencing another variable that is in vars
                let mut dep = self.vars.get(&a_cmp).unwrap().clone();
                dep.data_cmds.push(cmd);
                dep.sink_dims = sink_dims;
                self.vars.insert(
                    res_cmp.clone(),
                    dep
                );
            }
        }
    }

    pub fn print_alloc_tracker (&self) {
        println!("{}", self.alloc_tracker);
    }

    // wrapper over shape tracker
    pub fn get_shape (&self, id: &String) -> &Vec<usize> {
        self.shape_tracker.get_shape(id)
    }
}