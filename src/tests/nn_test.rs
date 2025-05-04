#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::autodiff;
    use crate::nn::{self, SeqF, Module};
    
    #[test]
    // #[ignore]
    fn nn_time_test () {
        autodiff::set_device(autodiff::devices::OpenCL::new());
        autodiff::eager_dep_opt();

        let mut neural_net = nn::Sequential();
        neural_net.insert(nn::Linear(256, 128, true));
        neural_net.insert(nn::Sigmoid());
        neural_net.insert(nn::Linear(128, 64, true));

        let mut opt = nn::opt::SGD(neural_net.params(), 0.1);

        let x = autodiff::randn(vec![2, 256]);
        let mut res = autodiff::empty();
        
        // prev: 0..1000        
        autodiff::ir_for(0..1, |_| {
            let y = neural_net.f(x.clone());
            opt.zero_grad();
            y.forward();
            y.backward();
            opt.step();

            res = y;
        });

        res.val().keep(); // ensure we can get in dependecy list

        let start = Instant::now();
        autodiff::print_and_exec();    
        let _ = res.val().get().round(4);

        println!("elapsed: {} s", start.elapsed().as_secs_f64());
    }
}
