import numpy as np
import os
from numpy.lib import ufunclike
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_runtime
import relay_utils

from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime


def run_tuning(tasks, task_weights, log_file):
    print("Begin tuning...")
    measure_runner = auto_scheduler.RPCRunner(
        "m1",
        "127.0.0.1",
        9190,
        min_repeat_ms=300,
        timeout=30,
        repeat=2
    )
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20000,
        runner=measure_runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    tuner.tune(tune_option)


if __name__ == "__main__":
    name = "bert-base-uncased"
    # The number of batches in an input.
    batch_size = 1
    # The length of each input sequence.
    seq_len = 128
    # target
    target = "metal"
    target_host = "llvm -mcpu=apple-latest -mtriple=arm64-apple-macos"
    # logfile
    log_file = "./assets/{name}_{target}".format(
        name=name.replace("/", "_"),
        target="metal"
    )
    print("Extract tasks...")
    mod, params, shape_dict = relay_utils.load_pt_model(name.replace("/", "_"))
    if not os.path.exists(log_file):
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"], params, target=target, target_host=target_host)
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" %
                  (idx, task.workload_key))
            print(task.compute_dag)

        run_tuning(tasks, task_weights, log_file)

    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target,
                              target_host=target_host, params=params)

    print("Upload")
    tmp = tempdir()
    filename = "net.tar"
    lib.export_library(tmp.relpath(filename))
    remote = auto_scheduler.utils.request_remote("m1", "127.0.0.1", 9190)
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    print("run")
    input_shape = [1, 128]
    dtype = "int64"
    ctx = remote.device(str(target), 0)
    module = runtime.graph_executor.GraphModule(rlib["default"](ctx))
    data_tvm = tvm.nd.array(
        (np.random.uniform(size=input_shape, low=0, high=10000)).astype(dtype))
    module.set_input("input_ids", data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)"
        % (np.mean(prof_res), np.std(prof_res))
    )
