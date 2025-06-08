use crate::{
    helper::shape::{global_to_ndim, ndim_to_global}, 
    kernel_decl::{Expression, Input, Matrix}, trackers::VarDependency
};
use crate::trackers::{
    DataCmds, 
    MatrixTracker,
    AccessType
};

impl<'a> MatrixTracker<'a> {
    // shared among get_mat and get_concat_dep
    pub fn ndim_change_datacmds (&self, ndim: &mut Vec<Expression>, var_dep: &VarDependency) {
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
                    *ndim = new_dim;
                },
                DataCmds::View { sink_dim, source_dim } => {
                    let global = ndim_to_global(ndim, sink_dim);
                    *ndim = global_to_ndim(
                        global,
                        source_dim
                    );
                },
            }
        }
    }

    pub fn get_mat (&self, id: &String, access_type: &AccessType) -> Matrix {
        if let Some(var_dep) = self.vars.get(id) {
            let sink_shape = &var_dep.sink_dims;
            let source_shape = &var_dep.source_dims;


            // go from global index --> N-dim index
            let mut ndim = match access_type {
                AccessType::Global => { 
                    global_to_ndim(
                        Expression::make_global(),
                        &sink_shape
                    )
                },
                AccessType::XY => {
                    vec![Expression::make_x(), Expression::make_y()]
                },
            };
            
            // manipulate the ndim expression thru data cmds in reverse
            self.ndim_change_datacmds(&mut ndim, var_dep); 

            // then, we can return the expression
            Matrix {
                id: var_dep.alloc_id.clone(),
                access: Expression::simplify( // simplify expression if needed
                    ndim_to_global(&ndim, source_shape)
                )
            }
        } 
        else if let Some(source_res) = self.sources.get(id) {
            match access_type {
                AccessType::Global => {
                    Matrix { 
                        id: source_res.alloc_id.clone(),
                        access: Expression::make_global()
                    }
                },
                AccessType::XY => {
                    let d = &source_res.dim;
                    assert!(
                        d.len() == 2, 
                        "Access type is XY, but the matrix isn't 2-dim"
                    );   

                    Matrix { 
                        id: source_res.alloc_id.clone(),
                        access: Expression::simplify(
                            ndim_to_global(
                                &vec![Expression::make_x(), Expression::make_y()],
                                d
                            )
                        )
                    }
                },
            }            

            
        } else {
            panic!("Unable to get matrix information on var {}", id);
        }
    }

    pub fn get_input (&self, id: &String, access_type: AccessType) -> Input {
        // you can probably cache this entire function...

        // check if constant
        if let Some(result) = self.get_concat(id, &access_type) {
            return result
        }
        
        // check if concat

        // else, check if normal matrix 
        Input::Mat { mat: self.get_mat(id, &access_type) } 
    }
}