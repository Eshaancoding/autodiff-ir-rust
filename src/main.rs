use std::time::Instant;
use autodiffv2::autodiff;
use autodiffv2::nn::{self, SeqF, Module};

pub fn main () {
    autodiff::set_device(autodiff::devices::CPUNew::new());
    let x = autodiff::randn(vec![5, 1]);
    let weight_matrix = x.broadcast(-1, 3).contigious(); 

    let input = autodiff::randn(vec![2, 5]);
    let result = autodiff::dot(input, weight_matrix);
    result.forward();
    result.val().keep();
    
    autodiff::print_and_exec();

    
}

// nn_test
pub fn tsdjfksjdf () {
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


// Transformer Test
pub fn sdjfksjdkf () {
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

pub fn tt () {
    autodiff::set_device(autodiff::devices::CPUNew::new());


    let x = autodiff::tensor(vec![
        0.03, 0.02, 0.01, 
        0.04, 0.05, 0.063, 
        0.01, 0.10, 0.07, 
        0.02, 0.01, 0.08, 
        0.05, 0.03, 0.06
    ], vec![5, 3]);

    let res = x.sum(-1);
    res.forward();
    res.val().keep();

    autodiff::print_and_exec();

    let v = res.val().get();
    println!("Value dim: {:#?}", v.dim);
    println!("Value data: {:#?}", v.data);
}