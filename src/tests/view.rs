#[cfg(test)]
mod tests {
    use crate::{autodiff, Tensor};

    #[test]
    fn view () {
        let a = autodiff::tensor(vec![2.0, 1.0, 3.0, 4.0], vec![2,2]);
        let res = a.view(1, 1, -1);
        res.val().keep();


    }
}