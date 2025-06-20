// gradient calculation, accumulation, and the add equal operation
#[cfg(test)]
mod tests {
    use crate::{autodiff, Tensor};
    
    fn f (a: &Tensor, b: &Tensor) -> Tensor {
        a.clone() * b.clone()
    }
    
    #[test]
    fn grd () {
        autodiff::set_device(autodiff::devices::CPU::new());

        autodiff::add_heading("Declaring tensors");
        let mut a = autodiff::tensor(vec![3.0, 2.0, 1.0, 3.0], vec![2, 2]);
        let b = autodiff::tensor(vec![2.0, 5.0, 2.0, 1.0], vec![2, 2]);

        autodiff::add_heading("Forward");
        let l = f(&a, &b);
        l.forward();
        autodiff::add_heading("Backprop");
        l.backward();

        a += a.grad(); // use .val() / .grad() to avoid recomputation

        let l = f(&a, &b);
        l.forward();
        autodiff::add_heading("Backprop v2");
        l.backward();
        
        autodiff::execute();      // actually executes operations from IR
        // autodiff::ir_print();

        let a_val = a.val().unwrap().get();
        assert_eq!(a_val.dim, vec![2, 2]);
        assert_eq!(a_val.data, vec![5.0, 7.0, 3.0, 4.0]);

        let grad_a = a.grad().get();  // tests the gradient accumulation from two backward movements
        assert_eq!(grad_a.dim, vec![2, 2]);
        assert_eq!(grad_a.data, vec![4.0, 10.0, 4.0, 2.0]);
    }
}