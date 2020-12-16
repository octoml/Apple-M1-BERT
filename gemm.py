import os

import numpy as np
from numpy.testing._private.utils import measure
import tvm
from tvm import te, auto_scheduler, topi


@auto_scheduler.register_workload
def gemm(M, N, K):
    a = te.placeholder((M, K), name="a")
    b = te.placeholder((N, K), name="b")
    k = te.reduce_axis((0, K), name="k")
    c = te.compute(
      (M, N),
      lambda i, j : te.sum(a[i, k] * b[j, k], [k]),
    )
    return [a, b, c]


@auto_scheduler.register_workload
def swish_gemm(M, N, K):
    a = te.placeholder((M, K), name="a")
    b = te.placeholder((N, K), name="b")
    k = te.reduce_axis((0, K), name="k")
    c = te.compute(
      (M, N),
      lambda i, j : te.sum(a[i, k] * b[j, k], [k]),
    )
    
    def hard_swish(x):
      x_plus_3 = te.compute(
        x.shape,
        lambda i, j: x[i, j] + 3.0
      )
      relu6 = tvm.topi.clip(x_plus_3, 0., 6.)
      return te.compute(
        x.shape,
        lambda i, j: relu6[i, j] * x[i, j] * 0.1666667
      )
      

    d = hard_swish(c)
    return [a, b, d]


if __name__ == "__main__":
    target = tvm.target.Target("metal --max_num_threads=1024")
    train_flag = True
    #target = tvm.target.Target("llvm -mcpu=apple-latest -mtriple=arm64-apple-darwin20.1.0")

    M, N, K = 128, 3072, 768
    task = auto_scheduler.SearchTask(func=gemm,
                                     args=(M, N, K),
                                     target=target)
    log_file = "gemm_{M}_{N}_{K}.json".format(M=M, N=N, K=K)
    # Inspect the computational graph
    print(task.compute_dag)
    

    if train_flag:
      measure_runner = auto_scheduler.RPCRunner("m1", "127.0.0.1", 9190, min_repeat_ms=300, timeout=30, repeat=3)
      tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        check_correctness=True,
        builder_n_parallel=1,
        runner=measure_runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
      )
      task.tune(tune_option)
      sch, args = task.apply_best(log_file)

      # Kill the process for measurement
      del measure_runner
    else:
      sch, args = task.apply_best(log_file)

    func = tvm.build(sch, args, target)

    # Check correctness
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(N, K)).astype(np.float32)
    c_np = np.dot(a_np, b_np.T)

    ctx = tvm.metal()
    #ctx = tvm.cpu()

    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    b_tvm = tvm.nd.array(b_np, ctx=ctx)
    c_tvm = tvm.nd.array(c_np, ctx=ctx)

    func(a_tvm, b_tvm, c_tvm)

    # Check results
    np.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-3)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500, number=400)
    ts = np.median(evaluator(a_tvm, b_tvm, c_tvm).results)
    print(
        "Execution time of this operator: %.3f ms"
        % (ts * 1000))
    print(
      "GFLOPS: %.3f" % (2 * M * N * K / ts / (10**9))
    )
