use crate::kernel_decl::{Matrix, Expression};
use crate::trackers::{AccessType, MatrixTracker};

/*
Difference between get_mat/get_input and get_res is that the id of the result might not be fully declared from the matrix tracker

Example:
    (0): a = mat(dim: [4, 32], contents: ...)
    (1): b = sum(a, dim=-1)

If we are on cmd (1), b hasn't been fully realized by the matrix tracker. It has only realized `a`.
However, we still need to know the dimensions

This also handles other edge cases
*/ 

impl<'a> MatrixTracker<'a> {
    
    pub fn get_res_xy (&self, id: &String, access_type: AccessType) -> Matrix {
        Matrix { id: "()".to_string(), access: Expression::make_x() }
    }
}