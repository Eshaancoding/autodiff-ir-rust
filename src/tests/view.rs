#[cfg(test)]
mod tests {
    use crate::autodiff;

    #[test]
    fn view () {
        autodiff::set_device(autodiff::devices::CPU::new());

        let a = autodiff::tensor(vec![2.0, 1.0, 3.0, 4.0], vec![2,2]);
        let res = a.view(vec![1, 1, 2, -1]);
        res.forward();
        res.val().keep();

        autodiff::execute();
        autodiff::ir_print(); 
    }
}