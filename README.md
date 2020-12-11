# Apple-M1-BERT

### Benchmark Apple TensorFlow with MLCompute
1. Install [Apple TensorFlow MacOS](https://github.com/apple/tensorflow_macos) on M1.
2. Install [Rust](https://www.rust-lang.org/tools/install)
3. Activate tf venv, install HuggingFace by `pip install transformers`
4. Dump `bert-base-uncased` model into a graph by running `python dump_tf.py`
4. Get CPU benchmark ("Graph Runtime: xxx ms.") by running `python run_tf.py cpu`
5. Get GPU benchmark ("Graph Runtime: xxx ms.") by running `python run_tf.py gpu`

### Running TVM AutoScheduler Search
We provide `search_dense_cpu.py` and `search_dense_gpu.py` for searching on M1 CPU and M1 GPU. Both scripts are using RPC. If you plan to run on M1 Mac Mini, run these commands in two terminal windows before running scripts.

Warning: Searching Metal schedule requires AutoScheduler correctness check feature, which is not merged into TVM main branch yet.
1. Start RPC Tracker: `python -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190`
2. Start RPC Server: `python -m tvm.exec.rpc_server --tracker 127.0.0.1:9190 --port 9090 --key m1 --no-fork`
3. Delete searched log in `./assets`, run scripts.

### Running TVM Inference
We provide pre-searched log from AutoScheduler in `assets` folder.Direct run `search_dense_cpu.py` and `search_dense_gpu.py` will be redirect to run inference test. Inference is also running on RPC Server.

For BERT model, we convert HuggingFace `transformers==3.0` Pytorch `bert-base-uncased` model to relay. It was done on a x86 machine and copied to M1. Model convert script can be found at `dump_pt.py`.

### Appendix: How to install TVM on ARM MacOS
We follow steps in https://discuss.tvm.apache.org/t/macos-m1-building-tvm-on-an-m1-mac/8494

A few things need to be noticed:
- Use python 3.8 rather than default 3.9 (`conda install python=3.8`)
- Build and install XGBoost==1.2.1 with similar command in the post

