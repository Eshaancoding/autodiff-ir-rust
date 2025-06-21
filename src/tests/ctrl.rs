// Control test:

#[cfg(test)]
mod tests {
    use crate::autodiff;
    
    #[test]
    fn if_ctrl () {
        autodiff::set_device(autodiff::devices::CPU::new());

        let x = autodiff::scalar(1.0);
        let mut y = autodiff::scalar(3.0);

        autodiff::ir_if(|| x, || {
            y += 4.0;
            y.forward();
        });

        y += 3.0;
        y.forward();

        let x_two = autodiff::scalar(0.0);
        autodiff::ir_if(|| x_two, || {
            y += 4.0;
            y.forward();
        });

        autodiff::print_and_exec();
        // autodiff::ir_print();
        
        let y_val = y.val().unwrap().get();
        assert_eq!(y_val.data, vec![10.0], "y data incorrect");
        assert_eq!(y_val.dim, vec![1], "y dim incorrect");
    }

    #[test]
    fn if_else_ctrl () {
        autodiff::set_device(autodiff::devices::CPU::new());

        let x = autodiff::scalar(1.0);
        let mut y = autodiff::scalar(3.0);
        let mut y_two = autodiff::scalar(3.0);

        autodiff::ir_if_else(|| x.clone(), || {
            y += 4.0;
            y.forward();
        }, || {
            y_two += 5.0;
            y_two.forward();
        });

        autodiff::ir_if_else(|| x - 1.0, || {
            y += 9.0;
            y.forward();
        }, || {
            y_two += 10.0;
            y_two.forward();
        });
        

        autodiff::execute();
        // autodiff::ir_print();
        
        let y_val = y.val().unwrap().get();
        assert_eq!(y_val.data, vec![7.0], "y data incorrect");
        assert_eq!(y_val.dim, vec![1], "y dim incorrect");

        let y_two_val = y_two.val().unwrap().get();
        assert_eq!(y_two_val.data, vec![13.0], "y_two data incorrect");
        assert_eq!(y_two_val.dim, vec![1], "y_two dim incorrect");
    }

    #[test]
    fn for_ctrl () {
        autodiff::set_device(autodiff::devices::CPU::new());
        
        let mut y = autodiff::scalar(10.0);
        autodiff::ir_for(-3..5, |i| {
            y += i.clone();
            y.forward();
        });

        autodiff::execute();
        // autodiff::ir_print();

        let y_val = y.val().unwrap().get();
        assert_eq!(y_val.data, vec![14.0], "y data incorrect");
        assert_eq!(y_val.dim, vec![1], "y dim incorrect");
    }

    // for loop and everything ctrl
    #[test] 
    fn evrty_ctrl () {
        autodiff::set_device(autodiff::devices::CPU::new());

        let mut y = autodiff::scalar(3.0);
        let mut y_two = autodiff::scalar(3.0);

        // no need to test ir_while --> if_for uses ir_while internally
        autodiff::ir_for(-3..5, |i| {
            autodiff::ir_if_else(
                || i.less_than(&autodiff::scalar(0.0)), 
                || {
                    y *= i.clone();
                    y.forward();
                }, 
                || {
                    autodiff::ir_if(
                        || i.more_than(&autodiff::scalar(0.0)), 
                        || {
                            y_two *= i.clone();
                            y_two.forward();
                        }
                    );
                }
            );
        });

        autodiff::execute();
        // autodiff::ir_print();

        let y_val = y.val().unwrap().get();
        assert_eq!(y_val.data, vec![-18.0], "y data incorrect");
        assert_eq!(y_val.dim, vec![1], "y dim incorrect");

        let y_two_val = y_two.val().unwrap().get();
        assert_eq!(y_two_val.data, vec![72.0], "y_two data incorrect");
        assert_eq!(y_two_val.dim, vec![1], "y_two dim incorrect");
    }
}