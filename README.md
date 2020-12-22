# Apple-M1-BERT Inference

### Update on Dec 21, 2020

By fixing `opt-level` bug in gpu inference script, we got a result on GPU inference. Now the result is:
```
% python search_dense_gpu.py 
Extract tasks...
Compile...
[23:27:38] /Users/bingxu/project/tvm/src/runtime/metal/metal_device_api.mm:138: Intializing Metal device 0, name=Apple M1
Upload
run
Evaluate inference time cost...
Mean inference time (std dev): 29.93 ms (0.19 ms)
```

### Benchmark Apple TensorFlow with MLCompute
1. Install [Apple TensorFlow MacOS](https://github.com/apple/tensorflow_macos) on M1.
2. Install [Rust](https://www.rust-lang.org/tools/install)
3. Activate tf venv, install HuggingFace by `pip install transformers`
4. Dump `bert-base-uncased` model into a graph by running `python dump_tf_graph.py`. 
5. Get Keras CPU benchmark by running `python run_keras.py --device cpu`. Sample output: `[Keras] Mean Inference time (std dev) on cpu: 579.0056343078613 ms (20.846548561801576 ms)`
6. Get Keras GPU benchmark by running `python run_keras.py --device gpu`. Sample output: `[Keras] Mean Inference time (std dev) on gpu: 1767.4337482452393 ms (27.00876036973127 ms)`
7. Get CPU benchmark by running `python run_graphdef.py --device cpu --graph-path ./models/bert-base-uncased.pb`. Sample output: `[Graphdef] Mean Inference time (std dev) on cpu: 512.3187007904053 ms (6.115432898641167 ms)`
8. Get GPU benchmark by running `python run_graphdef.py --device gpu --graph-path ./models/bert-base-uncased.pb`. Sample output: `[Graphdef] Mean Inference time (std dev) on gpu: 543.5237417221069 ms (4.210226676450006 ms)`

### Running TVM AutoScheduler Search
We provide `search_dense_cpu.py` and `search_dense_gpu.py` for searching on M1 CPU and M1 GPU. Both scripts are using RPC. If you plan to run on M1 Mac Mini, run these commands in two terminal windows before running scripts.

Warning: Searching Metal schedule requires AutoScheduler correctness check feature, which is not yet merged into TVM main branch yet. If you really want to try it today, you can use the unstable experimental branch at https://github.com/antinucleon/tvm/tree/metal

We will remove this warning once this feature is merged into main branch.

1. Start RPC Tracker: `python -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190`
2. Start RPC Server: `python -m tvm.exec.rpc_server --tracker 127.0.0.1:9190 --port 9090 --key m1 --no-fork`
3. Delete searched log in `./assets`, run scripts.

### Running TVM Inference
We provide pre-searched log from AutoScheduler in `assets` folder.Direct run `search_dense_cpu.py` and `search_dense_gpu.py` will be redirect to run inference test. Inference is also running on RPC Server.

For BERT model, we convert HuggingFace `transformers==3.0` Pytorch `bert-base-uncased` model to relay. It was done on a x86 machine and copied to M1. Model convert script can be found at `dump_pt.py`.

Sample output for `python search_dense_cpu.py`
```
Extract tasks...
Compile...
Upload
run
Evaluate inference time cost...
Mean inference time (std dev): 107.82 ms (3.39 ms)
```

Sample output for `python search_dense_gpu.py`
```
Extract tasks...
Compile...
-----------------------------------
Cannot find tuned schedules for target=metal -keys=metal,gpu -max_num_threads=256, workload_key=["ec4f7d9b3c9680b55f74f8646223586b"]. A fallback TOPI schedule is used, which may bring great performance regression or even compilation failure. Compute DAG info:
placeholder = PLACEHOLDER [1, 768]
placeholder = PLACEHOLDER [768, 768]
T_dense(i, j) += (placeholder[i, k]*placeholder[j, k])

[01:39:25] /Users/tvm/src/runtime/metal/metal_device_api.mm:138: Intializing Metal device 0, name=Apple M1
Upload
run
Evaluate inference time cost...
Mean inference time (std dev): 41.68 ms (0.34 ms)
```

### Why is TVM much faster than Apple TensorFlow with MLCompute?
- TVM AutoScheduler is able to using machine learning to search out CPU/GPU code optimization; Human experts are not able to cover all optimizations.
- TVM is able to fuse any subgraphs qualified of computation nature and directly generate code for the target; Human experts are only able to manually add fusing patterns, manually optimize certain subgraph.
- We visualized `bert-base-uncased` graph in Apple TensorFlow. Here is a sample block in BERT. ![sample block](assets/tf_block.png)
  As we can see, MLCompute tried to rewrite a TF graph, replace some operators to what it [supports](https://developer.apple.com/documentation/mlcompute/layers)
  In real practice perfect covert is alway hard, in BERT case, we can see `MatMul` operator is swapped to `MLCMatMul`, `LayerNorm` operator is swapped to `MLCLayerNorm`, while all others operators are not covered by MLCompute. In GPU case, data is copied between CPU and GPU almost in every step. On the other hand, TVM directly generates ALL operators on GPU, so it is able to maximize gpu utilization.


### Appendix: How to install TVM on ARM MacOS
We follow steps in https://discuss.tvm.apache.org/t/macos-m1-building-tvm-on-an-m1-mac/8494

A few things need to be noticed:
- Use python 3.8 rather than default 3.9 (`conda install python=3.8`)
- Build and install XGBoost==1.2.1 with similar command in the post

