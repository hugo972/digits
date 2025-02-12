use std::any::Any;
use serde::{Deserialize, Serialize};
use crate::nural::activation_fns::*;
use crate::nural::nural_network_layer::NuralNetworkLayer;
use crate::utils::matrix::Matrix;

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
        output_gradient: &[f64],
        _learning_rate: f64,
    ) -> Vec<f64> {
        Matrix::from(&output_gradient)
            .dot_mul(&Matrix::from(input).apply(self.activation_fn().dx))
            .data
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        input
            .iter()
            .map(|val| (self.activation_fn().fx)(*val))
            .collect::<Vec<f64>>()
    }
}