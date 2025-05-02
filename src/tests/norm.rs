#[cfg(test)]
mod tests {
    use crate::{autodiff, rms::RMS};
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
        
        res.val().keep();
        autodiff::execute();

        let data = res.val().get().round(4);    
        assert_eq!(data.dim, vec![2, 4]);
        assert_eq!(data.data, vec![
            0.3651, 0.7303, 1.0954, 1.4606,
            0.7581, 0.9097, 1.0613, 1.213
        ]);
    }
}
