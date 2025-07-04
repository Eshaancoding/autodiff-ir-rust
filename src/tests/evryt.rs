// everything test:
// Dot product, sum, & repeat (concat, unsqueeze), permute (transpose) test

#[cfg(test)]
mod tests {
    use crate::{autodiff, devices::CLDeviceType};

    #[test]
    fn everything () {
        autodiff::set_device(autodiff::devices::OpenCL::new(CLDeviceType::ALL));

        let x = autodiff::tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            vec![2, 4]
        );

        let y = autodiff::tensor(
            vec![7.0, 2.0, 6.0, 4.0, 9.0, 2.0, 6.0, 4.0],
            vec![2, 4]
        );

        let z = autodiff::tensor(
            vec![7.0, 2.0],
            vec![2, 1]
        );

        let v = autodiff::tensor(
            vec![1.0, 2.0, 4.0],
            vec![3, 1]
        );

        let t1 = autodiff::dot(x.clone(), y.t()); // 2, 2
        let t2 = t1.sum(0).repeat(|n, _| n.clone(), 3, 0); // 3, 2
        let t3 = autodiff::dot(t2, z.clone()); 
        let result = autodiff::concat(vec![t3, 3.0*v.pow2()], 1);
        result.forward();
        result.r(0..2, 0).backward(); // test r next

        result.val().unwrap().keep();

        autodiff::execute();
        autodiff::ir_print(); // ir print is massive

        // ======== Check resultant value ========
        let res_val = result.val().unwrap().get();
        assert_eq!(res_val.dim, vec![3,2], "Result value dim incorrect");
        assert_eq!(*res_val.data, vec![1518.0, 3.0, 1518.0, 12.0, 1518.0, 48.0], "Result value data incorrect");

        // ======== Check X grad ========
        let x_grad = x.grad().get();
        assert_eq!(x_grad.dim, vec![2, 4], "X grad dim incorrect");
        assert_eq!(*x_grad.data, vec![134.0, 36.0, 108.0, 72.0, 134.0, 36.0, 108.0, 72.0], "X grad data incorrect");

        // ======== Check Y grad ========
        let y_grad = y.grad().get();
        assert_eq!(y_grad.dim, vec![2, 4], "Y grad dim incorrect");
        assert_eq!(*y_grad.data, vec![84.0, 112.0, 140.0, 168.0, 24.0, 32.0, 40.0, 48.0], "Y grad data incorrect");

        // ======== Check Z grad ========
        let z_grad = z.grad().get();
        assert_eq!(z_grad.dim, vec![2, 1], "Y grad dim incorrect");
        assert_eq!(*z_grad.data, vec![332.0, 356.0], "Y grad data incorrect");

        // ======== Check V grad ========
        let v_grad = v.grad().get();
        assert_eq!(v_grad.dim, vec![3, 1], "Y grad dim incorrect");
        assert_eq!(*v_grad.data, vec![6.0, 12.0, 0.0], "Y grad data incorrect");
    }
}