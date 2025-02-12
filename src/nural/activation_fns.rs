use std::f64::consts::E;

pub struct ActivationFn<'a> {
    pub dx: &'a dyn Fn(f64) -> f64,
    pub fx: &'a dyn Fn(f64) -> f64,
}

pub const RELU: ActivationFn = ActivationFn {
    dx: &|x| if x > 0.0 { 1.0 } else { 0.0 },
    fx: &|x| if x > 0.0 { x } else { 0.0 },
};

pub const SIGMOID: ActivationFn = ActivationFn {
    dx: &|x| {
        let sigmoid = 1.0 / (1.0 + E.powf(-x));
        sigmoid * (1.0 - sigmoid)
    },
    fx: &|x| 1.0 / (1.0 + E.powf(-x)),
};

pub const TANH: ActivationFn = ActivationFn {
    dx: &|x| 1.0 - x.tanh().powf(2.0),
    fx: &|x| x.tanh(),
};
