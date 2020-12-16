import click
import time
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

def measure(func, args, repeats=50):
    res = []
    for _ in range(repeats):
        start = time.time()
        func(args)
        end = time.time()
        res.append((end - start) * 1000.)
    return np.mean(res), np.std(res)


def _load_keras_model(module, name, seq_len, batch_size):
    model = module.from_pretrained(name)
    dummy_input = tf.keras.Input(
        shape=[seq_len], batch_size=batch_size, dtype="int32")
    dummy_out = model(dummy_input)  # Propagate shapes through the keras model.
    return model

def keras_to_graphdef(model, batch_size, seq_len):
    model_func = tf.function(lambda x: model(x))
    input_dict = model._saved_model_inputs_spec
    input_spec = input_dict[list(input_dict.keys())[0]]
    model_func = model_func.get_concrete_function(
        tf.TensorSpec([batch_size, seq_len], input_spec.dtype)
    )
    frozen_func = convert_variables_to_constants_v2(model_func)
    return frozen_func.graph.as_graph_def()

def get_huggingface_model(name, batch_size, seq_len):
    import transformers
    module = getattr(transformers, "TFBertForSequenceClassification")
    model = _load_keras_model(
        module, name=name, batch_size=batch_size, seq_len=seq_len)
    return model

