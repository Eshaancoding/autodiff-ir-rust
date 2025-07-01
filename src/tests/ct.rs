// constant & elw & funcs test

#[cfg(test)]
mod tests {
    use crate::autodiff;

    #[test]
    fn ct () {
        autodiff::set_device(autodiff::devices::CPU::new());

        let x = autodiff::tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            vec![2, 4]
        );

        let y = autodiff::tensor(
            vec![7.0, 2.0, 6.0, 4.0, 9.0, 2.0, 6.0, 4.0],
            vec![2, 4]
        );

        // t = cos(2x + 3)
        // m = sqrt((t*y) + 3(t + x))
        // y = t / m
        let t = (2.0 * x.clone() + 3.0).cos(); 
        let t2 =  ((t.clone() * y.clone()) + 3.0 * (t.clone() + x.clone())).sqrt();
        let res = t2 / t;

        res.forward();
        res.backward();

        res.val().unwrap().keep(); 
        autodiff::execute();

        // ======== Check resultant value ========
        let res_val = res.val().unwrap().get().round(4);
        assert_eq!(res_val.dim, vec![2, 4], "Result value dim incorrect");
        assert_eq!(*res_val.data, vec![
            8.5169,   4.1459,  -0.9816, 783.7341,
            5.6071,  -4.9606, -15.6412,   5.6242
        ], "Result data value incorrect");

        // ======== Check x value ========
        let x_grad = x.grad().get().round(4);
        assert_eq!(x_grad.dim, vec![2, 4], "x grad value dim incorrect");
        assert_eq!(*x_grad.data, vec![
            -41.4012,      6.4684,      3.5990, -353617.0418,
            4.4253,      9.1042,   -117.8712,      1.7871
        ], "y grad value incorrect");
        
        // ======== Check y value ========
        let y_grad = y.grad().get().round(4);
        assert_eq!(y_grad.dim, vec![2, 4], "x grad dim incorrect");
        assert_eq!(*y_grad.data, vec![
            0.2070, 0.1600, 0.5591, 0.1442,
            0.0983, 0.1327, 0.1162, 0.0899
        ], "y grad value incorrect");
    }
}