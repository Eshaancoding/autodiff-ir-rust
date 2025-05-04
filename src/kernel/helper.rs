use crate::decl::{Input, Matrix};

use super::decl::Kernels;

// get the needed input matrixes for a kernel
// this is necessary to find the number of matrixes to enter into the kernel function
pub fn get_input_ids (kernel: &Kernels) -> Vec<String> {
    fn get_matrix_ids (a: &Matrix) -> Vec<String> {
        vec![a.id.clone()]
    }

    fn get_input_ids (a: &Input) -> Vec<String> {
        match a {
            Input::ConcatMatrix { id_one, id_two, .. } => {
                vec![
                    get_input_ids(id_one),
                    get_input_ids(id_two),
                ].concat()
            },
            Input::Mat { mat } => {
                get_matrix_ids(mat)
            },
            _ => { vec![] }
        }
    }

    match kernel {
        Kernels::Binary { a, b, res, .. } => {
            vec![
                get_input_ids(a),
                get_input_ids(b),
                get_matrix_ids(res)
            ].concat()
        },
        Kernels::DotProd { a, b, res } => {
            vec![
                get_input_ids(a),
                get_input_ids(b),
                get_matrix_ids(res)
            ].concat()
        },
        Kernels::Movement { a, res } => {
            vec![
                get_input_ids(a),
                get_matrix_ids(res)
            ].concat()
        },
        Kernels::Reduce { a, res, .. } => {
            vec![
                get_input_ids(a),
                get_matrix_ids(res)
            ].concat()
        },
        Kernels::Unary { a, res, .. } => {
            vec![
                get_input_ids(a),
                get_matrix_ids(res)
            ].concat()
        }
    }
}