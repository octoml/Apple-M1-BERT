import tf_utils
import numpy as np
import click
import os
import tensorflow as tf
import time

@click.command()
@click.option("--graph-path", required=True)
@click.option('--device', required=True, help="device will be used, [cpu] or [gpu]")
def main(graph_path, device):
    if device == "cpu":
        tf.config.experimental.set_visible_devices([], 'GPU')
    elif device == "gpu":
        tf.device('/gpu:0')
    else:
        raise Exception("Unknown devices")    
    batch_size = 1
    seq_len = 128
    if os.path.exists(graph_path) is False:
        raise Exception("Graph doesn't exist. Please dump tf graph first.")
    with tf.io.gfile.GFile(graph_path, "rb") as fi:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(fi.read())
    g = tf.graph_util.import_graph_def(graph_def)

    with tf.compat.v1.Session(graph=g) as sess:
        dummy_input = np.random.randint(0, 10000, size=[batch_size, seq_len]).astype(np.int32)
        x = tf.compat.v1.get_default_graph().get_tensor_by_name('x:0')
        fetches = ["tf_bert_for_sequence_classification/classifier/BiasAdd:0"]
        feed_dict = {x: dummy_input}
        # warm up
        _ = sess.run(fetches=["tf_bert_for_sequence_classification/classifier/BiasAdd:0"],
                                feed_dict={x: dummy_input})
        def run_graph(args):
            sess = args[0]
            fetches = args[1]
            feed_dict = args[2]
            _ = sess.run(fetches=["tf_bert_for_sequence_classification/classifier/BiasAdd:0"],
                                feed_dict={x: dummy_input})
        run_args = [
            sess,
            fetches,
            feed_dict
        ]
        mean, std = tf_utils.measure(run_graph, run_args)
    print("[Graphdef] Mean Inference time (std dev) on {device}: {mean_time} ms ({std} ms)".format(
        device=device, mean_time=mean, std=std
    ))

if __name__ == "__main__":
    main()