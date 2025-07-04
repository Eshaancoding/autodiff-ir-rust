use std::time::Instant;
use autodiffv2::{autodiff, opencl_matmul, opencl_reduce};
use autodiffv2::devices::{OpenCL, CLDeviceType};
use autodiffv2::nn::{self, SeqF, Module};

// in the future, probably migrate to tests()

// nn_test
pub fn nn_test () {
    autodiff::set_device(OpenCL::new(CLDeviceType::GPU));
    autodiff::eager_dep_opt();

    let mut neural_net = nn::Sequential();
    neural_net.insert(nn::Linear(128, 64, true));
    neural_net.insert(nn::Sigmoid());
    neural_net.insert(nn::Linear(64, 32, true));

    let mut opt = nn::optimizers::SGD(neural_net.params(), 0.01);

    let x = autodiff::randn(vec![2, 128]);
    let mut res = autodiff::empty();

    // prev: 0..1000        
    autodiff::ir_for(0..10, |_| {
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
    println!("id to be read: {}", res.val().unwrap().id);
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
            0.03, 0.023, 0.623, 0.0123, 0.7792, 0.3727, 0.137, 0.062,
            0.23, 0.013, 0.683, 0.023, 0.292, 0.327, 0.198, 0.022
        ], 
        vec![2, 8]
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

    nn_test();
    // multihead_att();

    // forward();
    // me_life();
    // broadcasting_test();
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