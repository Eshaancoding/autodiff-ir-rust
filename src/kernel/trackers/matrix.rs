/*
=== Matrix Tracker ===
Given an id, extract's the source matrix id, the expression needed to go to id.
Also, handles constants the concatenation matrixes + constants
This is how memory is handled within all backend kernels.
*/

use std::collections::HashMap;
use crate::{decl::{Expression, Input, Matrix}, ir::optimizations::helper::ir_to_res, IRCmds};
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
    alloc_id: String,
    source_dims: Vec<usize>,
    sink_dims: Vec<usize>,
    data_cmds: Vec<DataCmds>
}

#[derive(Clone, Debug)]
pub struct VarSource {
    alloc_id: String,
    dim: Vec<usize>
}

pub struct MatrixTracker<'a> {
    // given sink variable, get source variable and the steps to reach to sink var
    vars: HashMap<String, VarDependency>,  
    sources: HashMap<String, VarSource>,            // tracks source variables (no var dependency)
    shape_tracker: ShapeTracker,                     // tracks the shape of variables
    alloc_tracker: &'a AllocTracker<'a>
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
            // a_cmp = a.clone();
            // res_cmp = res.clone();
            // concat is weird, not doing this yet
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
                if self.vars.contains_key(&id) {
                    self.vars.remove_entry(&id);
                }

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

        if let Some(cmd) = data_cmd {
            let sink_dims = self.shape_tracker.get_shape(&res_cmp).clone();

            if res_cmp == a_cmp { // referencing itself, so append to the list
                if self.vars.contains_key(&res_cmp) {
                    let var_m = self.vars.get_mut(&res_cmp).unwrap();
                    var_m.data_cmds.push(cmd);
                    var_m.sink_dims = sink_dims;
                }
                else {
                    let var_source = self.sources.get(&a_cmp).unwrap().clone();
                    self.vars.insert(
                        res_cmp.clone(),
                        VarDependency {
                            alloc_id: var_source.alloc_id,
                            source_dims: var_source.dim,
                            data_cmds: vec![cmd],
                            sink_dims,
                        }
                    );
                    self.sources.remove_entry(&res_cmp);
                }
            }
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
            }
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

    pub fn global_to_ndim (index:Expression, shape: &Vec<usize>) -> Vec<Expression> {
        let n = shape.len();
        let mut strides: Vec<Expression> = vec![Expression::make_const(1); n];
        for i in (0..(n - 1)).rev() {
            strides[i] = Expression::make_const(strides[i+1].get_const().unwrap() * shape[i+1] as i32);
        }

        let nd_index: Vec<Expression> = (0..n)
            .map(|i| 
                Expression::make_remainder(
                    Expression::make_div(
                        index.clone(),
                        strides[i].clone()
                    ), 
                    Expression::make_const(shape[i] as i32)
                )
            )
            .collect();

        nd_index
    }

    pub fn ndim_to_global (dim: Vec<Expression>, shape: &Vec<usize>) -> Expression {
        let mut global_expr = Expression::make_mult(
            dim[0].clone(),
            Expression::make_const(shape[0] as i32)
        );

        for i in 1..shape.len() {
            global_expr = Expression::make_add(
                global_expr,
                Expression::make_mult(
                    dim[i].clone(), 
                    Expression::make_const(shape[i] as i32) 
                )
            );
        }

        global_expr
    }

    pub fn get_input (&self, id: String) -> Input {
        if self.vars.contains_key(&id) {
            let var_dep = self.vars.get(&id).unwrap();
            let sink_shape = &var_dep.sink_dims;
            let source_shape = &var_dep.source_dims;

            // go from global index --> N-dim index
            let mut ndim = MatrixTracker::global_to_ndim(
                Expression::make_global(),
                &sink_shape
            );
            
            // manipulate the ndim expression thru data cmds in reverse
            for cmd in var_dep.data_cmds.iter().rev() {
                match cmd { 
                    DataCmds::Broadcast { dim, .. } => {
                        ndim[*dim] = Expression::make_const(0);
                    },
                    DataCmds::Index { index, dim } => {
                        ndim[*dim] = Expression::make_const(*index as i32);
                    },
                    DataCmds::Permute { p } => {
                        let mut new_dim = vec![Expression::make_const(0); ndim.len()];
                        for i in 0..ndim.len() {
                            new_dim[i] = ndim[p[i]].clone()
                        }
                        ndim = new_dim;
                    },
                    DataCmds::View { sink_dim, source_dim } => {
                        ndim = MatrixTracker::global_to_ndim(
                            MatrixTracker::ndim_to_global(ndim, sink_dim),
                            source_dim
                        );
                    },
                    DataCmds::Concat => {} // TODO!
                }
            }

            // then, we can return the expression
            Input::Mat { 
                mat: Matrix {
                    id: var_dep.alloc_id.clone(),
                    access: MatrixTracker::ndim_to_global(ndim, source_shape),
                }
            }
        } 
        else if self.sources.contains_key(&id) {
            let source_res = self.sources.get(&id).unwrap().clone();

            Input::Mat { 
                mat: Matrix { 
                    id: source_res.alloc_id,
                    access: Expression::make_global()
                }
            }
        } else {
            panic!("Unable to get matrix information on var {}", id);
        }
    }

    pub fn print (&self) {
        println!("Sources: {:#?}", self.sources);
        println!("Vars: {:#?}", self.vars);
    }
}