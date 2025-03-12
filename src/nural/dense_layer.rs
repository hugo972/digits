use crate::nural::nural_network_layer::NuralNetworkLayer;
use ndarray::Array2;
use rand::Rng;
use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::any::Any;
use std::fmt;

pub struct DenseLayer {
    bias: Array2<f64>,
    weights: Array2<f64>,
}

impl DenseLayer {
    pub fn new(inputs: usize, outputs: usize) -> DenseLayer {
        let mut rng = rand::rng();
        DenseLayer {
            bias: Array2::from_shape_fn((outputs, 1), |_| rng.random_range(-1.0..1.0)),
            weights: Array2::from_shape_fn((outputs, inputs), |_| rng.random_range(-1.0..1.0)),
        }
    }
}

impl NuralNetworkLayer for DenseLayer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn backward(
        &mut self,
        input: &[f64],
        _output: &[f64],
        output_gradient: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        let output_gradient_vec =
            Array2::from_shape_vec((output_gradient.len(), 1), output_gradient.to_vec()).unwrap();
        let input_vec = Array2::from_shape_vec((input.len(), 1), input.to_vec()).unwrap();
        let weights_gradient_mx = output_gradient_vec.dot(&input_vec.t());
        let input_gradient_mx = self.weights.t().dot(&output_gradient_vec);

        self.weights = &self.weights - &weights_gradient_mx * learning_rate;
        self.bias = &self.bias - output_gradient_vec * learning_rate;

        let (input_gradient, _) = input_gradient_mx.into_raw_vec_and_offset();
        input_gradient
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let input_vec = Array2::from_shape_vec((input.len(), 1), input.to_vec()).unwrap();
        let (output, _) = (&self.weights.dot(&input_vec) + &self.bias).into_raw_vec_and_offset();
        output
    }
}

#[derive(Deserialize, Serialize)]
struct DenseLayerData {
    bias: Vec<f64>,
    bias_shape: [usize; 2],
    weights: Vec<f64>,
    weights_shape: [usize; 2],
}

impl Serialize for DenseLayer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let (bias, _) = self.bias.clone().into_raw_vec_and_offset();
        let (weights, _) = self.weights.clone().into_raw_vec_and_offset();
        let data = DenseLayerData {
            bias,
            bias_shape: self.bias.shape()[0..=1].try_into().unwrap(),
            weights,
            weights_shape: self.weights.shape()[0..=1].try_into().unwrap(),
        };

        serializer.serialize_newtype_struct("Data", &data)
    }
}

impl<'a> Deserialize<'a> for DenseLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        struct DataVisitor;
        impl<'a> Visitor<'a> for DataVisitor {
            type Value = DenseLayerData;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("Data")
            }

            fn visit_newtype_struct<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'a>,
            {
                let data = DenseLayerData::deserialize(deserializer)?;
                Ok(data)
            }
        }

        let data = deserializer.deserialize_newtype_struct("Data", DataVisitor)?;
        Ok(DenseLayer {
            bias: Array2::from_shape_vec(data.bias_shape, data.bias).unwrap(),
            weights: Array2::from_shape_vec(data.weights_shape, data.weights).unwrap(),
        })
    }
}
