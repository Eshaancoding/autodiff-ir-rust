use crate::Tensor;

// just need params for optimizer update
pub trait Module {
    fn params (&self) -> Vec<Tensor>;
}

pub trait SeqF : Module {
    fn f (&self, x: Tensor) -> Tensor;
}