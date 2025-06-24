use crate::{
    helper::shape::{global_to_ndim, ndim_to_global}, 
    kernel_decl::{Expression, Input, Matrix}
};
use crate::trackers::{
    MatrixTracker,
    AccessType
};

impl MatrixTracker {    
    pub fn get_inp_dep (&self, id: &String, ndim: &mut Vec<Expression>) -> Input {
        // is vars concat
        if let Some(result) = self.vars_concat.get(id) {
            self.ndim_change_datacmds(ndim, &result.data_cmds);

            let mut ndim_two = ndim.clone();
            ndim_two[result.source.dim] = Expression::make_minus(
                ndim_two[result.source.dim].clone(),
                Expression::make_const(result.source.idx_end as i32)
            );

            Input::ConcatMatrix { 
                id_one: Box::new(self.get_inp_dep(&result.source.a, ndim)), 
                id_two: Box::new(self.get_inp_dep(&result.source.b, &mut ndim_two)), 
                conditional: Expression::simplify(
                    Expression::make_more_than(ndim[result.source.dim].clone(), Expression::make_const(result.source.idx_end as i32 - 1))
                )
            }
        }
        // is sources concat
        else if let Some(result) = self.sources_concat.get(id) {
            let mut ndim_two = ndim.clone();
            ndim_two[result.dim] = Expression::make_minus(
                ndim_two[result.dim].clone(),
                Expression::make_const(result.idx_end as i32)
            );

            Input::ConcatMatrix { 
                id_one: Box::new(self.get_inp_dep(&result.a, ndim)), 
                id_two: Box::new(self.get_inp_dep(&result.b, &mut ndim_two)), 
                conditional: Expression::simplify(
                    Expression::make_more_than(ndim[result.dim].clone(), Expression::make_const(result.idx_end as i32 - 1))
                )
            }
        } 

        // dependency on var
        else if let Some(var_dep) = self.vars.get(id) {
            self.ndim_change_datacmds(ndim, &var_dep.data_cmds);

            Input::Mat { 
                mat: Matrix { 
                    id: var_dep.id.clone(), 
                    access: Expression::simplify(
                        ndim_to_global(ndim, &var_dep.source_dims)
                    ) 
                }
            }
        } 
        
        // on source var
        else if let Some(source_res) = self.sources.get(id) {
            Input::Mat { 
                mat: Matrix { 
                    id: source_res.id.clone(), 
                    access: Expression::simplify(
                        ndim_to_global(ndim, &source_res.dim)
                    )
                }
            }
        }

        // is constant
        else if let Some(content) = self.constant_tracker.get_f64(id) {
            Input::Constant { val: content }
        }
        
        else {
            panic!("Invalid matrix type!");
        }
    }

    pub fn get_input (&self, id: &String, access_type: AccessType) -> Input {
        let sink_shape = self.shape_tracker.get_shape(id);

        let mut ndim = match access_type {
            AccessType::Global => { 
                global_to_ndim(
                    Expression::make_global(),
                    &sink_shape
                )
            },
            AccessType::XY => {
                assert_eq!(sink_shape.len(), 2, "Access type XY but concat result has result of 2");
                vec![Expression::make_x(), Expression::make_y()]
            },
        };

        // short circuit if referencing source (it's nicer to display)
        // global_to_ndim(ndim_to_global(x)) doesn't quite simplify to x yet
        if let Some(source_res) = self.sources.get(id) {
            match access_type {
                AccessType::Global => {
                    Input::Mat { mat: Matrix { 
                        id: source_res.id.clone(),
                        access: Expression::make_global()
                    } }
                },
                AccessType::XY => {
                    let d = &source_res.dim;
                    assert!(
                        d.len() == 2, 
                        "Access type is XY, but the matrix isn't 2-dim"
                    );   

                    Input::Mat { mat: Matrix { 
                        id: source_res.id.clone(),
                        access: Expression::simplify(
                            ndim_to_global(
                                &vec![Expression::make_x(), Expression::make_y()],
                                d
                            )
                        )
                    } }
                },
            }            

        } 
        else {
            self.get_inp_dep(id, &mut ndim)
        }
    }
}