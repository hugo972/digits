use crate::nural::activation_fns::*;
use crate::nural::nural_network_layer::NuralNetworkLayer;
use serde::{Deserialize, Serialize};
use std::any::Any;

#[derive(Deserialize, Serialize)]
pub struct ActivationLayer {
    kind: ActivationLayerKind,
}

#[derive(Deserialize, Serialize)]
pub enum ActivationLayerKind {
    ReLu,
    Sigmoid,
    Tanh,
}

impl ActivationLayer {
    pub fn new(kind: ActivationLayerKind) -> ActivationLayer {
        ActivationLayer { kind }
    }

    fn activation_fn(&self) -> ActivationFn {
        match self.kind {
            ActivationLayerKind::ReLu => RELU,
            ActivationLayerKind::Sigmoid => SIGMOID,
            ActivationLayerKind::Tanh => TANH,
        }
    }
}

impl NuralNetworkLayer for ActivationLayer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn backward(
        &mut self,
        input: &[f64],
        _output: &[f64],
        output_gradient: &[f64],
        _learning_rate: f64,
    ) -> Vec<f64> {
        input.iter()
            .zip(output_gradient.iter())
            .map(|(input_val, output_gradient_val)| (self.activation_fn().dx)(*input_val) * *output_gradient_val)
            .collect()
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        input
            .iter()
            .map(|val| (self.activation_fn().fx)(*val))
            .collect::<Vec<f64>>()
    }
}
