#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::autodiff;
    use crate::nn::{self, SeqF, Module};
    
    #[test]
    fn nn () {
        autodiff::set_irbuilder(autodiff::TensorRsIRBuilder::new());
        autodiff::eager_dep_opt();

        let mut neural_net = nn::Sequential();
        neural_net.insert(nn::Linear(256, 128, true));
        neural_net.insert(nn::Sigmoid());
        neural_net.insert(nn::Linear(128, 64, true));

        
        // usually you would define a linear func like this
        // neural_net.insert(nn::Linear(3, 2, true)); 

        let mut opt = nn::opt::SGD(neural_net.params(), 0.1);

        let x = autodiff::randn(vec![2, 256]);
        let mut res = autodiff::empty_tensor();
        
        autodiff::ir_for(0..100, |_| {
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

        res.val().keep(); // ensure we can get in dependecy list

        
        let start = Instant::now();
        autodiff::execute();    
        // autodiff::ir_print();
        let _ = res.val().get().round(4);
        println!("elapsed: {} s", start.elapsed().as_secs_f64());
    }
}
