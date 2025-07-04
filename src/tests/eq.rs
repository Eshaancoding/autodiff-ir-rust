#[cfg(test)]

mod tests {
    use crate::{autodiff, devices::CLDeviceType, Tensor};

    #[test]
    fn eq () {
        autodiff::set_device(autodiff::devices::OpenCL::new(CLDeviceType::ALL));

        // ============ Equality test ============ 
        let x = autodiff::tensor(
            vec![0.0, 1.0, 3.0, 2.0, 1.0, 3.0],
            vec![6]
        );

        let res: Tensor = 3.0 * 
            (
                x.clone() * (x.equal(&autodiff::constant(3.0, x.dim())))
            ).pow2();
            
        let y = res.sum(0);

        autodiff::add_heading("Forward");
        y.forward();
        autodiff::add_heading("Backward");
        y.backward();

        // ============ all no test ============ 
        let all_no = autodiff::tensor(
            vec![0.0, 1.0, 1.0],
            vec![3]
        ).all();

        autodiff::add_heading("All No Forward");
        all_no.forward(); 

        // ============ all yes test ============ 
        let all_yes = autodiff::tensor(
            vec![1.0, 1.0, 1.0],
            vec![3]
        ).all();

        autodiff::add_heading("All Yes Forward");
        all_yes.forward(); 

        // ============ Execute ============ 
        res.val().unwrap().keep();
        all_no.val().unwrap().keep();
        all_yes.val().unwrap().keep();
        autodiff::print_and_exec();

        // ============ Check ============ 
        let res_val = res.val().unwrap().get();
        assert_eq!(res_val.dim, vec![6], "Res dim incorrect");
        assert_eq!(*res_val.data, vec![0.0, 0.0, 27.0, 0.0, 0.0, 27.0], "Res data incorrect");

        let grad_val = x.grad().get();
        assert_eq!(grad_val.dim, vec![6], "Grad dim incorrect");
        assert_eq!(*grad_val.data, vec![0.0, 0.0, 18.0, 0.0, 0.0, 18.0], "Grad value incorrect");

        let all_no = all_no.val().unwrap().get();
        assert_eq!(all_no.dim, vec![1]);
        assert_eq!(*all_no.data, vec![0.0]);

        let all_yes = all_yes.val().unwrap().get();
        assert_eq!(all_yes.dim, vec![1]);
        assert_eq!(*all_yes.data, vec![1.0]);
    }
}