use crate::{
    helper::shape::{global_to_ndim, ndim_to_global}, 
    kernel_decl::{Expression, Input, Matrix}
};
use crate::trackers::{
    MatrixTracker,
    AccessType
};

impl<'a> MatrixTracker<'a> {    
    pub fn get_concat_dep (&self, id: &String, ndim: &mut Vec<Expression>) -> Input {
        if let Some(result) = self.vars_concat.get(id) {
            let mut ndim_two = ndim.clone();
            ndim_two[result.dim] = Expression::make_minus(
                ndim_two[result.dim].clone(),
                Expression::make_const(result.idx_end as i32)
            );

            Input::ConcatMatrix { 
                id_one: Box::new(self.get_concat_dep(&result.a, ndim)), 
                id_two: Box::new(self.get_concat_dep(&result.b, &mut ndim_two)), 
                conditional: Expression::simplify(
                    Expression::make_more_than(ndim[result.dim].clone(), Expression::make_const(result.idx_end as i32 - 1))
                )
            }
        } 
        else if let Some(var_dep) = self.vars.get(id) {
            self.ndim_change_datacmds(ndim, var_dep);

            Input::Mat { 
                mat: Matrix { 
                    id: var_dep.alloc_id.clone(), 
                    access: Expression::simplify(
                        ndim_to_global(ndim, &var_dep.source_dims)
                    ) 
                }
            }
        } 
        else if let Some(source_res) = self.sources.get(id) {
            Input::Mat { 
                mat: Matrix { 
                    id: source_res.alloc_id.clone(), 
                    access: Expression::simplify(
                        ndim_to_global(ndim, &source_res.dim)
                    )
                }
            }
        }
        else {
            panic!("Invalid dependency for concat! Can't be a constant");
        }
    }

    pub fn get_concat (&self, id: &String, access_type: &AccessType) -> Option<Input> {
        if let Some(result) = self.vars_concat.get(id) {
            let mut ndim = match access_type {
                AccessType::Global => { 
                    global_to_ndim(
                        Expression::make_global(),
                        &result.sink_dim
                    )
                },
                AccessType::XY => {
                    assert_eq!(result.sink_dim.len(), 2, "Access type XY but concat result has result of 2");
                    vec![Expression::make_x(), Expression::make_y()]
                },
            };            

            Some(self.get_concat_dep(id, &mut ndim))
        } else {
            None
        }
    }
}