﻿use crate::nural::loss_fns::{LossFn, BINARY_CROSS_ENTROPY, MSE};
use crate::nural::nural_network_layer::NuralNetworkLayer;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

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

    pub fn load_file(file_path: &str) -> Result<NuralNetwork, std::io::Error> {
        let serialized_bytes = std::fs::read(file_path)?;
        Ok(serde_cbor::from_slice::<NuralNetwork>(&serialized_bytes).unwrap())
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        self.forward(input).last().unwrap().clone()
    }

    pub fn save_file(&self, file_path: &str) -> Result<(), std::io::Error> {
        let serialized_bytes = serde_cbor::to_vec(self).unwrap();
        std::fs::write(file_path, serialized_bytes)
    }

    pub fn train(&mut self, data: &[(Vec<f64>, Vec<f64>)], epochs: usize) {
        for epoch in 0..epochs {
            let mut error = 0.0;

            for (input, expected_output) in data.iter() {
                let outputs = self.forward(input);
                let output = outputs.last().unwrap();

                error += (self.loss_fn().fx)(&output, &expected_output);

                let mut gradient = (self.loss_fn().dx)(&output, &expected_output);
                for (layer_index, layer) in self.layers.iter_mut().enumerate().rev() {
                    gradient = layer.backward(
                        &outputs[layer_index],
                        &outputs[layer_index + 1],
                        &gradient,
                        self.learning_rate,
                    );
                }
            }

            println!(
                "epoch {}/{} error: {}",
                epoch + 1,
                epochs,
                error / data.len() as f64
            );
        }
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

impl Display for NuralNetwork {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let serialized_nural_network = serde_json::to_string_pretty(&self).unwrap();
        write!(f, "Nural Network: {}", serialized_nural_network)
    }
}
