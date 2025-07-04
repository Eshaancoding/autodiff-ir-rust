use std::time::Instant;
use autodiffv2::autodiff;
use autodiffv2::devices::{OpenCL, CLDeviceType};
use autodiffv2::nn::{self, SeqF, Module};

// in the future, probably migrate to tests()

// nn_test
pub fn nn_test () {
    autodiff::set_device(OpenCL::new(CLDeviceType::GPU));
    autodiff::eager_dep_opt();

    let mut neural_net = nn::Sequential();
    neural_net.insert(nn::Linear(512, 256, true));
    neural_net.insert(nn::Sigmoid());
    neural_net.insert(nn::Linear(256, 128, true));
    neural_net.insert(nn::Sigmoid());

    let mut opt = nn::optimizers::SGD(neural_net.params(), 0.01);

    let x = autodiff::randn(vec![16, 512]);
    let mut res = autodiff::empty();

    // prev: 0..1000       
    autodiff::ir_for(0..10000, |_| {
        let y = neural_net.f(x.clone());
        opt.zero_grad();
        y.forward();
        y.backward();
        opt.step();

        res = y;
    });

    x.grad().keep();
    res.val().unwrap().keep(); // ensure we can get in dependency list

    autodiff::print_and_exec();    
    let v = res.val().unwrap().get().round(4);
    println!("v data len: {}", v.data.len());
    println!("first value: {}", v.data[0]);
    println!("second value: {}", v.data[1]);
    println!("third value: {}", v.data[2]);
}


// Transformer Test
pub fn multihead_att () {
    autodiff::set_device(OpenCL::new(CLDeviceType::GPU));

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

pub fn simple () {
    autodiff::set_device(OpenCL::new(CLDeviceType::GPU));

    let a = autodiff::tensor(
        vec![
            2.0, 0.0, 1.0
        ], 
        vec![3]
    );
    let res = a.sum(0);

    res.forward();
    res.val().unwrap().keep();

    autodiff::print_and_exec();

    let v = res.val().unwrap().get();
    println!("value dim: {:#?}", v.dim);
    println!("value data: {:#?}", v.data);
    println!("value id: {:#?}", v.id);
}

pub fn main () {
    // simple();    
    // opencl_matmul();
    // opencl_reduce();
    // println!("hello world");

    // multihead_att();
    nn_test();
    


    // test constant tracker
    // check whether it deletes vars if += or -= and if it deletes references vars too
    // check if it keeps constants

    // test concat tracker
    // check whether you can do concat --> view --> permute --> concat --> permute --> concat
    // nested structure like that 
}