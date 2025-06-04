use crate::kernel_decl::{Expression, Input, Matrix};
use super::trackers::{DataCmds, MatrixTracker};

// Different access types (depending on the kernel used) is needed
#[derive(PartialEq, Eq)]
pub enum AccessType {
    Global,         // Elementwise; uses Global IDX
    XY,             // Dot product; restricted to matrix 2-dim; uses X and Y. (SEE src/matmul_cpu/512_matmul.cpp for example of x + y)
    Dim {d: usize}  // Reduce; Select one dim and iterate it over; uses Global Idx
}

impl<'a> MatrixTracker<'a> {
    pub fn calc_stride (shape: &Vec<usize>) -> Vec<Expression> {
        let n = shape.len();
        let mut strides: Vec<Expression> = vec![Expression::make_const(1); n];
        for i in (0..(n - 1)).rev() {
            strides[i] = Expression::make_const(strides[i+1].get_const().unwrap() * shape[i+1] as i32);
        }

        strides
    }

    pub fn global_to_ndim (index:Expression, shape: &Vec<usize>) -> Vec<Expression> {
        let strides = MatrixTracker::calc_stride(shape);

        let nd_index: Vec<Expression> = (0..shape.len())
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
        let strides = MatrixTracker::calc_stride(shape);
        let mut global_expr = Expression::make_mult(
            dim[0].clone(),
            strides[0].clone()
        );

        for i in 1..shape.len() {
            global_expr = Expression::make_add(
                global_expr,
                Expression::make_mult(
                    dim[i].clone(), 
                    strides[i].clone() 
                )
            );
        }

        global_expr
    }

    pub fn get_mat (&self, id: &String, access_type: AccessType) -> Matrix {
        if let Some(var_dep) = self.vars.get(id) {
            let sink_shape = &var_dep.sink_dims;
            let source_shape = &var_dep.source_dims;

            // go from global index --> N-dim index
            let mut ndim = match access_type {
                AccessType::Global => { 
                    MatrixTracker::global_to_ndim(
                        Expression::make_global(),
                        &sink_shape
                    )
                },
                AccessType::XY => {
                    assert!(
                        var_dep.sink_dims.len() == 2,
                        "Access type is XY, but the matrix isn't 2-dim"
                    );
                    vec![Expression::make_x(), Expression::make_y()]
                },
                _ => todo!()
            };
            
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
                        let global = MatrixTracker::ndim_to_global(ndim, sink_dim);
                        ndim = MatrixTracker::global_to_ndim(
                            global,
                            source_dim
                        );
                    },
                    DataCmds::Concat => {} // TODO!
                }
            }

            // then, we can return the expression
            Matrix {
                id: var_dep.alloc_id.clone(),
                access: Expression::simplify( // simplify expression if needed
                    MatrixTracker::ndim_to_global(ndim, source_shape)
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
                            MatrixTracker::ndim_to_global(
                                vec![Expression::make_x(), Expression::make_y()],
                                d
                            )
                        )
                        
                    }
                },
                AccessType::Dim { d } => todo!()
            }            

            
        } else {
            panic!("Unable to get matrix information on var {}", id);
        }
    }

    pub fn get_input (&self, id: &String, access_type: AccessType) -> Input {
        // check if constant
        // check if concat

        Input::Mat { mat: self.get_mat(id, access_type) } 
    }

    // for debugging
    pub fn print_raw (&self, id: &String) {
        if self.sources.contains_key(id) {
            println!("{:#?}", self.sources.get(id).unwrap())
        } 
        else if self.vars.contains_key(id) {
            println!("{:#?}", self.vars.get(id).unwrap())
        }
        else {
            println!("id not found");
        }
    }
}