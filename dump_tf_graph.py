import tensorflow as tf
import tf_utils
import click
import os

def save_model_pb(graphdef, name, prefix="./models"):
    tf.io.write_graph(graphdef,
                      prefix,
                      name.replace('/', '_') + ".pb",
                      False)


@click.command()
@click.option('--model-name', default='bert-base-uncased', help='name of model')
@click.option('--save-prefix', default='./models')
def main(model_name, save_prefix):
    batch_size = 1
    seq_len = 128
    model = tf_utils.get_huggingface_model(model_name, batch_size, seq_len)
    graphdef = tf_utils.keras_to_graphdef(model, batch_size, seq_len)
    save_model_pb(graphdef, model_name, save_prefix)


if __name__ == '__main__':
    main()