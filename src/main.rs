use std::time::Instant;
use autodiffv2::autodiff;
use autodiffv2::nn::{self, SeqF, Module};

// in the future, probably migrate to tests()

pub fn concat_test () {
    autodiff::set_device(autodiff::devices::CPUNew::new());
    // -1 dim
    let input = autodiff::randn(vec![2, 5]);
    let w_one = autodiff::randn(vec![5, 1]);
    let w_two = autodiff::randn(vec![5, 1]);
    let t = autodiff::concat(vec![w_one, w_two], -1);
    let result = autodiff::dot(input, t);
    result.forward();
    result.val().unwrap().keep();

    // 0 dim
    let input = autodiff::randn(vec![5, 2]);
    let w_one = autodiff::randn(vec![1,2]);
    let w_two = autodiff::randn(vec![1,2]);
    let t = autodiff::concat(vec![w_one, w_two], 0);
    let result = autodiff::dot(input, t);
    result.forward();
    result.val().unwrap().keep();

    autodiff::print_and_exec();
}

pub fn broadcasting_test () {
    autodiff::set_device(autodiff::devices::CPUNew::new());
    
    let input = autodiff::randn(vec![5, 25]);
    
    let res = input.t().contigious();
    res.forward();
    res.val().unwrap().keep();
    
    autodiff::print_and_exec();
}

// nn_test
pub fn nn_test () {
    autodiff::set_device(autodiff::devices::CPUNew::new());

    autodiff::eager_dep_opt();

    let mut neural_net = nn::Sequential();
    neural_net.insert(nn::Linear(256, 128, true));
    neural_net.insert(nn::Sigmoid());
    neural_net.insert(nn::Linear(128, 64, true));

    let mut opt = nn::optimizers::SGD(neural_net.params(), 0.1);

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

    res.val().unwrap().keep(); // ensure we can get in dependency list

    let start = Instant::now();
    autodiff::print_and_exec();    
    let _ = res.val().unwrap().get().round(4);

    println!("elapsed: {} s", start.elapsed().as_secs_f64());
}


// Transformer Test
pub fn multihead_att () {
    autodiff::set_device(autodiff::devices::CPUNew::new());

    let transformer = nn::MultiHeadAttention(64, 4);
    let mut opt = nn::optimizers::SGD(transformer.params(), 0.01);

    let x = autodiff::randn(vec![32, 64]);
    let res = transformer.f(x.clone(), x.clone(), x.clone());
    res.forward();
    res.backward();
    opt.step();
    
    // x.grad().keep();
    res.val().unwrap().keep(); // ensure we can get in dependecy list

    let start = Instant::now();
    autodiff::print_and_exec();    
    let _ = res.val().unwrap().get().round(4);

    println!("elapsed: {} s", start.elapsed().as_secs_f64());
}

pub fn reduce () {
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
    res.val().unwrap().keep();

    autodiff::print_and_exec();

    let v = res.val().unwrap().get();
    println!("Value dim: {:#?}", v.dim);
    println!("Value data: {:#?}", v.data);
}

pub fn forward () {
    autodiff::set_device(autodiff::devices::CPU::new());

    let mut x = autodiff::tensor(vec![3.0], vec![1]);
    
    autodiff::ir_for(2..6, |i| {
        x += i.clone(); 
        x.forward();
    });    

    let result = x * 3.0;
    result.forward();

    autodiff::print_and_exec();

    let v = result.val().unwrap().get();
    println!("value dim: {:#?}", v.dim);
    println!("value data: {:#?}", v.data);
    println!("value id: {:#?}", v.id);
}

pub fn main () {
    // forward();
    // me_life();
    nn_test();
    // broadcasting_test();
    // multihead_att();
    // reduce();
    // concat_test();
    // broadcasting_test();

    // test constant tracker
    // check whether it deletes vars if += or -= and if it deletes references vars too
    // check if it keeps constants

    // test concat tracker
    // check whether you can do concat --> view --> permute --> concat --> permute --> concat
    // nested structure like that 
}