use crate::kernel_decl::{Expression, Matrix, Output};


impl Output {
    pub fn id (&self) -> &String {
        match self {
            Output::Mat { mat } => { &mat.id },
            Output::Temp => panic!("Calling id() of temp")
        }
    }

    pub fn mut_id (&mut self) -> &mut String {
        match self {
            Output::Mat { mat } => { &mut mat.id },
            Output::Temp => panic!("Calling id() of temp")
        }
    }

    pub fn access (&self) -> &Expression {
        match self {
            Output::Mat { mat } => { &mat.access },
            Output::Temp => panic!("Calling access() of temp")
        }
    }

    pub fn get_mut_mat (&mut self) -> Option<&mut Matrix> {
        match self {
            Output::Mat { mat } => Some(mat),
            _ => None
        }
    }
}