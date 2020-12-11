import time
import sys
import tensorflow as tf
import numpy as np


def run_graph_model(frozen_graph, batch_size, seq_len):
    with tf.io.gfile.GFile(frozen_graph, "rb") as fi:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(fi.read())
    g = tf.graph_util.import_graph_def(graph_def)
    repeats = 50
    with tf.compat.v1.Session(graph=g) as sess:
        dummy_input = np.random.randint(0, 10000, size=[batch_size, seq_len]).astype(np.int32)
        x = tf.compat.v1.get_default_graph().get_tensor_by_name('x:0')
        _ = sess.run(fetches=["tf_bert_for_sequence_classification/classifier/BiasAdd:0"],
                                feed_dict={x: dummy_input})

        start = time.time()
        for i in range(repeats):
            _ = sess.run(fetches=["tf_bert_for_sequence_classification/classifier/BiasAdd:0"],
                                feed_dict={x: dummy_input})
        end = time.time()
    print("Graph Runtime: %f ms." % (1000 * ((end - start) / repeats)))


if __name__ == "__main__":
    from tensorflow.python.compiler.mlcompute import mlcompute
    if sys.argv[1] == "cpu":
        mlcompute.set_mlc_device(device_name='cpu')
    elif sys.argv[1] == "gpu":
        mlcompute.set_mlc_device(device_name='gpu')
    else:
        raise Exception()
    assert len(sys.argv) == 2
    name = "bert-base-uncased"
    # The number of batches in an input.
    batch_size = 1
    # The length of each input sequence.
    seq_len = 128
    run_graph_model("./models/" + name + ".pb", batch_size, seq_len)