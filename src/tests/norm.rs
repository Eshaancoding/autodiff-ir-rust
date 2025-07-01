#[cfg(test)]
mod tests {
    use crate::{autodiff, LayerNorm, RMS};
    use crate::nn::module::SeqF;

    #[test]
    pub fn rmsnorm () {
        autodiff::set_device(autodiff::devices::CPU::new());

        let x = autodiff::tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4]
        );

        let rms_layer = RMS(4);
        let res = rms_layer.f(x);
        res.forward();
        
        res.val().unwrap().keep();
        autodiff::execute();

        let data = res.val().unwrap().get().round(4);    
        assert_eq!(data.dim, vec![2, 4]);
        assert_eq!(*data.data, vec![
            0.3651, 0.7303, 1.0954, 1.4606,
            0.7581, 0.9097, 1.0613, 1.213
        ]);
    }

    #[test]
    pub fn layernorm () {
        autodiff::set_device(autodiff::devices::CPU::new());

        let x = autodiff::tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0],
            vec![2, 4]
        );

        let layer_norm = LayerNorm(4);
        let res = layer_norm.f(x);
        res.forward();
        
        res.val().unwrap().keep();
        autodiff::execute();

        let data = res.val().unwrap().get().round(4);    
        assert_eq!(data.dim, vec![2, 4]);
        assert_eq!(*data.data, vec![
            -1.3416, -0.4472, 0.4472, 1.3416,
            -1.1832, -0.5071, 0.169, 1.5213
        ]);
    }
}
