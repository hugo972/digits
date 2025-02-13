pub struct LossFn<'a> {
    pub dx: &'a dyn Fn(&[f64], &[f64]) -> Vec<f64>,
    pub fx: &'a dyn Fn(&[f64], &[f64]) -> f64,
}

pub const BINARY_CROSS_ENTROPY: LossFn = LossFn {
    dx: &|actual, expected| {
        actual
            .iter()
            .zip(expected.iter())
            .map(|(actual_val, expected_val)| {
                ((1.0 - actual_val) / (1.0 - expected_val) - actual_val / expected_val)
                    / actual.len() as f64
            })
            .collect()
    },
    fx: &|actual, expected| {
        let sum =
            actual
                .iter()
                .zip(expected.iter())
                .fold(0.0, |val, (&actual_val, &expected_val)| {
                    val - actual_val * expected_val.log(10.0)
                        - (1.0 - actual_val) * (1.0 - expected_val).log(10.0)
                });
        sum / (actual.len() as f64)
    },
};

pub const MSE: LossFn = LossFn {
    dx: &|actual, expected| {
        actual
            .iter()
            .zip(expected.iter())
            .map(|(actual_val, expected_val)| {
                (actual_val - expected_val) * 2.0 / actual.len() as f64
            })
            .collect()
    },
    fx: &|actual, expected| {
        actual
            .iter()
            .zip(expected.iter())
            .fold(0.0, |val, (&actual_val, &expected_val)| {
                val + (expected_val - actual_val).powi(2)
            })
            / (actual.len() as f64)
    },
};
