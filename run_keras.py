import tf_utils
import numpy as np
import click


@click.command()
@click.option("==model-name", default='bert-base-uncased')
@click.option('--device', required=True, help="device will be used, [cpu] or [gpu]")
def main(model_name, device):
    if device == "cpu":
        tf.config.experimental.set_visible_devices([], 'GPU')
    elif device == "gpu":
        tf.device('/gpu:0')
    else:
        raise Exception("Unknown devices")    
    batch_size = 1
    seq_len = 128
    model = tf_utils.get_huggingface_model(model_name,
                                           batch_size,
                                           seq_len)
    def run_keras(args):
        assert len(args) == 2
        model = args[0]
        np_input = args[1]
        _ = model(np_input)
    
    run_args = [
        model,
        np.random.randint(0, 10000, size=[batch_size, seq_len]).astype(np.int32)
    ]

    mean, std = tf_utils.measure(run_keras, run_args)
    print("[Keras] Mean Inference time (std dev) on {device}: {mean_time} ms ({std} ms)".format(
        device=device, mean_time=mean, std=std
    ))

if __name__ == "__main__":
    main()