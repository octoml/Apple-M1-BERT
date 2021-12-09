import numpy as np
import coremltools as ct
import tensorflow as tf
import time
from timeit import default_timer as timer
from coremltools.models.neural_network import quantization_utils

from transformers import TFBertForSequenceClassification, BertTokenizer


print('---Getting Model---')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

max_seq_length = 128
input_shape = (1, max_seq_length) #(batch_size, maximum_sequence_length)

input_layer = tf.keras.layers.Input(shape=input_shape[1:], dtype=tf.int64, name='input')

prediction_model = bert_model(input_layer)
tf_model = tf.keras.models.Model(inputs=input_layer, outputs=prediction_model)

print('---Converting Model to CoreML---')

mlmodel = ct.convert(tf_model)#, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_GPU)


# Fill the input with zeros to adhere to input_shape
input_values = np.zeros(input_shape)
# Store the tokens from our sample sentence into the input
input_values[0,:8] = np.array(tokenizer.encode("Hello, my dog is cute")).astype(np.int64)

print('---Running Predictions---')

num_iters = 1000
start = timer()
for i in range(num_iters):
    results = mlmodel.predict({'input':input_values}) # 'input' is the name of our input layer from (3)
end = timer()
print("avg latency: {} ms".format(((end - start) * 1000) / num_iters))

