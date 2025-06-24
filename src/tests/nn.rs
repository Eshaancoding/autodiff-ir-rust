#[cfg(test)]
mod tests {
    use crate::autodiff;
    use crate::nn::{self, SeqF, Module};
    
    #[test]
    fn nn () {
        autodiff::set_device(autodiff::devices::CPU::new());
        autodiff::eager_dep_opt();

        let l1_w = autodiff::tensor(vec![
            0.03, 0.02, 0.01, 
            0.04, 0.05, 0.063, 
            0.01, 0.10, 0.07, 
            0.02, 0.01, 0.08, 
            0.05, 0.03, 0.06
        ], vec![5, 3]);

        let l1_b = autodiff::tensor(vec![
            0.03, 0.01, 0.032
        ], vec![3]);

        let l2_w = autodiff::tensor(vec![
            0.022, 0.016, 
            0.007, 0.093, 
            0.013, 0.09
        ], vec![3, 2]);

        let mut neural_net = nn::Sequential();
        neural_net.insert(nn::LinearWithWeights(l1_w.clone(), Some(l1_b.clone())));
        neural_net.insert(nn::Sigmoid());
        neural_net.insert(nn::LinearWithWeights(l2_w.clone(), None));
        
        // usually you would define a linear func like this
        // neural_net.insert(nn::Linear(3, 2, true)); 

        let mut opt = nn::optimizers::SGD(neural_net.params(), 0.1);

        let x = autodiff::tensor(vec![
            0.01, 0.02, 0.037, 0.04, 0.05,
            0.063, 0.07, 0.08, 0.09, 0.013
        ], vec![2, 5]);

        let mut res = autodiff::empty();
        
        autodiff::ir_for(0..10, |_| {
            let y = neural_net.f(x.clone());
            opt.zero_grad();

            autodiff::add_heading("Forward");
            y.forward();

            autodiff::add_heading("Backward");
            y.backward();

            autodiff::add_heading("Stepping");
            opt.step();

            res = y;
        });

        res.val().unwrap().keep(); // ensure we can get in dependecy list

        autodiff::execute();    
        autodiff::ir_print(); // even more massive

        let res_out_data = res.val().unwrap().get().round(4);
        assert_eq!(res_out_data.dim, vec![2,2], "Y output dim wrong");
        assert_eq!(res_out_data.data, vec![
            -1.6599, -1.5678,
            -1.6662, -1.5736
        ], "Y output data wrong");
        
        let l1_w_out_data = l1_w.val().unwrap().get().round(4);
        assert_eq!(l1_w_out_data.dim, vec![5,3], "L1 Weight dim wrong");
        assert_eq!(l1_w_out_data.data, vec![
            0.0461, 0.0349, 0.025,
            0.0599, 0.0683, 0.0815,
            0.0358, 0.1238, 0.094,
            0.0487, 0.0365, 0.1067,
            0.0639, 0.0428, 0.0729
        ], "l1_w output data wrong");

        let l1_b_out_data = l1_b.val().unwrap().get().round(4);
        assert_eq!(l1_b_out_data.dim, vec![3], "L1 Bias dim wrong");
        assert_eq!(l1_b_out_data.data, vec![
            0.4715, 0.4172, 0.4423
        ], "l1_b output data wrong");

        let l2_w_out_data = l2_w.val().unwrap().get().round(4);
        assert_eq!(l2_w_out_data.dim, vec![3, 2], "L2 Weight dim wrong");
        assert_eq!(l2_w_out_data.data, vec![
            -1.0534, -1.0594,
            -1.0530, -0.9670,
            -1.0603, -0.9833
        ], "l2_w output data wrong");
    }
}
