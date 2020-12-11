import time
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def load_keras_model(module, name, seq_len, batch_size, report_runtime=True):
    model = module.from_pretrained(name)
    dummy_input = tf.keras.Input(
        shape=[seq_len], batch_size=batch_size, dtype="int32")
    dummy_out = model(dummy_input)  # Propagate shapes through the keras model.
    start = time.time()
    repeats = 50
    for i in range(repeats):
        dummy_out = model(dummy_input)
    end = time.time()
    print("Keras Runtime: %f ms." % (1000 * ((end - start) / repeats)))
    return model


def convert_to_graphdef(model, batch_size, seq_len):
    model_func = tf.function(lambda x: model(x))
    input_dict = model._saved_model_inputs_spec
    input_spec = input_dict[list(input_dict.keys())[0]]
    model_func = model_func.get_concrete_function(
        tf.TensorSpec([batch_size, seq_len], input_spec.dtype)
    )
    frozen_func = convert_variables_to_constants_v2(model_func)
    return frozen_func.graph.as_graph_def()

def download_model(name, batch_size, seq_len):
    import transformers

    module = getattr(transformers, "TFBertForSequenceClassification")
    model = load_keras_model(
        module, name=name, batch_size=batch_size, seq_len=seq_len)
    return convert_to_graphdef(model, batch_size, seq_len)


if __name__ == "__main__":
    # The name of the transformer model to download and run.
    #name = "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad"
    name = "bert-base-uncased"
    # The number of batches in an input.
    batch_size = 1
    # The length of each input sequence.
    seq_len = 128
    model = download_model(name, batch_size, seq_len)
    tf.io.write_graph(model, "./models", name + ".pb", False)
