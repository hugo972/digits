pub struct LossFn<'a> {
    pub dx: &'a dyn Fn(&[f64], &[f64]) -> Vec<f64>,
    pub fx: &'a dyn Fn(&[f64], &[f64]) -> f64,
}

pub const BINARY_CROSS_ENTROPY: LossFn = LossFn {
    dx: &|predicted, actual| {
        actual
            .iter()
            .zip(predicted.iter())
            .map(|(actual_val, predicted_val)| {
                ((1.0 - actual_val) / (1.0 - predicted_val) - actual_val / predicted_val)
                    / actual.len() as f64
            })
            .collect()
    },
    fx: &|predicted, actual| {
        let sum =
            actual
                .iter()
                .zip(predicted.iter())
                .fold(0.0, |val, (&actual_val, &predicted_val)| {
                    val - actual_val * predicted_val.log(10.0)
                        - (1.0 - actual_val) * (1.0 - predicted_val).log(10.0)
                });
        sum / (actual.len() as f64)
    },
};

pub const MSE: LossFn = LossFn {
    dx: &|predicted, actual| {
        actual
            .iter()
            .zip(predicted.iter())
            .map(|(actual_val, predicted_val)| {
                (predicted_val - actual_val) * 2.0 / actual.len() as f64
            })
            .collect()
    },
    fx: &|predicted, actual| {
        let sum = actual
            .iter()
            .zip(predicted.iter())
            .fold(0.0, |val, (&actual_val, &predicted_val)| {
                val + (actual_val - predicted_val).powi(2)
            });
        sum / (actual.len() as f64)
    },
};
