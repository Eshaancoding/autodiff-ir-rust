use crate::helper::shape::ndim_to_global;
use crate::kernel_decl::{Expression, Matrix, Output};
use crate::trackers::{AccessType, KernelTracker};

/*
The accessing method of the result matrix will always be a global or x & y.

Example:
    (0): a = mat(dim: [4, 32], contents: ...)
    (1): b = sum(a, dim=-1)

If we are on cmd (1), b hasn't been fully realized by the matrix tracker. It has only realized `a`.
However, we still need to know the dimensions. This is calculated manually without using shape tracker
*/ 

impl KernelTracker {
    pub fn get_res (&self, id: &String, access_type: AccessType, expected_shape: &Vec<usize>) -> Output {
        return match access_type {
            AccessType::XY => {
                Output::Mat { mat: Matrix { 
                    id: id.clone(), 
                    access: Expression::simplify(ndim_to_global(
                        &vec![Expression::make_x(), Expression::make_y()], 
                        &expected_shape
                    ))
                }
            } },
            AccessType::Global => {
                Output::Mat { mat: 
                    Matrix { 
                        id: id.clone(), 
                        access: Expression::make_global() 
                    }
                }
            }
        }
    }
}