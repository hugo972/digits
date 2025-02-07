use crate::nural::nural_network::NuralNetworkLayer;
use crate::utils::matrix::Matrix;

pub struct TransformLayer {
    bias: Matrix,
    weights: Matrix,
}

impl TransformLayer {
    pub fn new(inputs: usize, outputs: usize) -> TransformLayer {
        TransformLayer {
            bias: Matrix::rnd(outputs, 1),
            weights: Matrix::rnd(outputs, inputs),
        }
    }
}

impl NuralNetworkLayer for TransformLayer {
    fn backward(&mut self, input: &[f64], output_gradient: &[f64], learning_rate: f64) -> Vec<f64> {
        let output_gradient_mx = Matrix::from(&output_gradient).transpose();
        let weights_gradient_mx = output_gradient_mx.mul(&Matrix::from(&input));
        let input_gradient_mx = self.weights.transpose().mul(&output_gradient_mx);

        self.weights = self.weights.sub(&weights_gradient_mx.scale(learning_rate));
        self.bias = self.bias.sub(&output_gradient_mx.scale(learning_rate));

        input_gradient_mx.data
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .mul(&Matrix::from(&input).transpose())
            .add(&self.bias)
            .data
    }
}