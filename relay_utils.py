import os
import tvm
from tvm import relay

def load_pt_model(name,
                  path="./models/",
                  relay_file="_pt_model.json",
                  relay_params="_pt_model.params"):
    with open(os.path.join(path, name + relay_file), "r") as fi:
        mod = tvm.ir.load_json(fi.read())
    with open(os.path.join(path, name + relay_params), "rb") as fi:
        params = relay.load_param_dict(fi.read())
    mod = tvm.relay.transform.FastMath()(mod)
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, ctx:
                            tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
    mod = BindPass(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    return mod, params, {"input_ids" : [1, 128]}


