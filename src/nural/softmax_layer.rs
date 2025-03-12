use crate::nural::nural_network_layer::NuralNetworkLayer;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::any::Any;

#[derive(Deserialize, Serialize)]
pub struct SoftmaxLayer;

impl SoftmaxLayer {
    pub fn new() -> SoftmaxLayer {
        SoftmaxLayer
    }
}

impl NuralNetworkLayer for SoftmaxLayer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn backward(
        &mut self,
        _input: &[f64],
        output: &[f64],
        output_gradient: &[f64],
        _learning_rate: f64,
    ) -> Vec<f64> {
        let output_vec = Array2::from_shape_vec((output.len(), 1), output.to_vec()).unwrap();
        let output_gradient_vec = Array2::from_shape_vec((output_gradient.len(), 1), output_gradient.to_vec()).unwrap();
        let identity = Array2::ones((output.len(), 1));
        let (output, _) = ((identity - output_vec.t()) * output_vec).dot(&output_gradient_vec).into_raw_vec_and_offset();
        output
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let exp_sum = input.iter().map(|val| (*val).exp()).sum::<f64>();
        input
            .iter()
            .map(|val| (*val).exp() / exp_sum)
            .collect::<Vec<f64>>()
    }
}
