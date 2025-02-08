use crate::nural::loss_fns::{LossFn, BINARY_CROSS_ENTROPY, MSE};
use crate::nural::nural_network_layer::NuralNetworkLayer;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct NuralNetwork {
    layers: Vec<Box<dyn NuralNetworkLayer>>,
    learning_rate: f64,
    loss_kind: NuralNetworkLossKind,
}

#[derive(Deserialize, Serialize)]
pub enum NuralNetworkLossKind {
    BinaryCrossEntropy,
    Mse,
}

impl NuralNetwork {
    pub fn new(
        layers: Vec<Box<dyn NuralNetworkLayer>>,
        learning_rate: f64,
        loss_kind: NuralNetworkLossKind,
    ) -> Self {
        NuralNetwork {
            layers,
            learning_rate,
            loss_kind,
        }
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        self.forward(input).last().unwrap().clone()
    }

    pub fn train(&mut self, data: &[(Vec<f64>, Vec<f64>)], epochs: usize) -> f64 {
        let mut error = 0.0;

        for epoch in 0..epochs {
            for (input, expected_output) in data.iter() {
                let outputs = self.forward(input);
                let output = outputs.last().unwrap();

                if epoch == epochs - 1 {
                    error += (self.loss_fn().fx)(&output, &expected_output);
                }

                let mut gradient = (self.loss_fn().dx)(&output, &expected_output);

                for (layer, input) in self
                    .layers
                    .iter_mut()
                    .rev()
                    .zip(outputs.iter().rev().skip(1))
                {
                    gradient = layer.backward(input, &gradient, self.learning_rate);
                }
            }
        }

        error / data.len() as f64
    }

    fn forward(&self, input: &[f64]) -> Vec<Vec<f64>> {
        let mut outputs = vec![input.to_vec(); 1];
        for layer in self.layers.iter() {
            let output = layer.forward(outputs.last().unwrap());
            outputs.push(output);
        }
        outputs
    }

    fn loss_fn(&self) -> LossFn {
        match self.loss_kind {
            NuralNetworkLossKind::BinaryCrossEntropy => BINARY_CROSS_ENTROPY,
            NuralNetworkLossKind::Mse => MSE,
        }
    }
}
