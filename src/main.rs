use std::time::Instant;
use autodiffv2::autodiff;
use autodiffv2::nn::{self, SeqF, Module};

// for testing; 
// same as nn_test

/*
pub fn main () {
    autodiff::set_device(autodiff::devices::CPUNew::new());

    autodiff::eager_dep_opt();

    let mut neural_net = nn::Sequential();
    neural_net.insert(nn::Linear(256, 128, true));
    neural_net.insert(nn::Sigmoid());
    neural_net.insert(nn::Linear(128, 64, true));

    let mut opt = nn::opt::SGD(neural_net.params(), 0.1);

    let x = autodiff::randn(vec![2, 256]);
    let mut res = autodiff::empty();
    
    // prev: 0..1000        
    autodiff::ir_for(0..100, |_| {
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
*/

pub fn main () {
    autodiff::set_device(autodiff::devices::CPUNew::new());

    let transformer = nn::MultiHeadAttention(64, 4);
    let mut opt = nn::opt::SGD(transformer.params(), 0.01);

    let x = autodiff::randn(vec![32, 64]);
    let res = transformer.f(x.clone(), x.clone(), x.clone());
    res.forward();
    res.backward();
    opt.step();
    
    // x.grad().keep();
    res.val().keep(); // ensure we can get in dependecy list

    let start = Instant::now();
    autodiff::print_and_exec();    
    let _ = res.val().get().round(4);

    println!("elapsed: {} s", start.elapsed().as_secs_f64());
}