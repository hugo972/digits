use crate::nural::activation_layer::ActivationLayer;
use crate::nural::dense_layer::DenseLayer;
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::any::{Any, TypeId};
use std::fmt;
use crate::nural::softmax_layer::SoftmaxLayer;

pub trait NuralNetworkLayer {
    fn as_any(&self) -> &dyn Any;
    fn backward(&mut self, input: &[f64], output: &[f64], output_gradient: &[f64], learning_rate: f64) -> Vec<f64>;
    fn forward(&self, input: &[f64]) -> Vec<f64>;
}

impl Serialize for Box<dyn NuralNetworkLayer> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let layer = self.as_any();
        let layer_type_id = layer.type_id();
        if layer_type_id == TypeId::of::<ActivationLayer>() {
            let mut state = serializer.serialize_struct("Layer", 2)?;
            state.serialize_field("type", "ActivationLayer")?;
            state.serialize_field("data", layer.downcast_ref::<ActivationLayer>().unwrap())?;
            state.end()
        } else if layer_type_id == TypeId::of::<DenseLayer>() {
            let mut state = serializer.serialize_struct("Layer", 2)?;
            state.serialize_field("type", "DenseLayer")?;
            state.serialize_field("data", layer.downcast_ref::<DenseLayer>().unwrap())?;
            state.end()
        } else if layer_type_id == TypeId::of::<SoftmaxLayer>() {
            let mut state = serializer.serialize_struct("Layer", 2)?;
            state.serialize_field("type", "SoftmaxLayer")?;
            state.serialize_field("data", layer.downcast_ref::<SoftmaxLayer>().unwrap())?;
            state.end()
        } else {
            Err(serde::ser::Error::custom("Unknown Layer type"))
        }
    }
}

impl<'a> Deserialize<'a> for Box<dyn NuralNetworkLayer> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        struct LayerVisitor;

        impl<'a> Visitor<'a> for LayerVisitor {
            type Value = Box<dyn NuralNetworkLayer>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("serialize_struct::Layer")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Box<dyn NuralNetworkLayer>, M::Error>
            where
                M: MapAccess<'a>,
            {
                let layer_type = match map.next_key::<String>()? {
                    Some(layer_type_key) => {
                        assert_eq!(layer_type_key, "type");
                        map.next_value::<String>()?
                    }
                    None => panic!("Missing layer type"),
                };

                map.next_key::<String>()?.expect("Missing layer data");

                match layer_type.as_str() {
                    "ActivationLayer" => map
                        .next_value::<ActivationLayer>()
                        .map(|l| Box::new(l) as Box<dyn NuralNetworkLayer>),
                    "DenseLayer" => map
                        .next_value::<DenseLayer>()
                        .map(|l| Box::new(l) as Box<dyn NuralNetworkLayer>),
                    "SoftmaxLayer" => map
                        .next_value::<SoftmaxLayer>()
                        .map(|l| Box::new(l) as Box<dyn NuralNetworkLayer>),
                    _ => Err(de::Error::custom("Unknown Layer type")),
                }
            }
        }

        deserializer.deserialize_struct("Layer", &["type", "data"], LayerVisitor)
    }
}
