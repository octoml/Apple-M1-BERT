import torch
import transformers  # pip3 install transfomers==3.0
import tvm
from tvm import relay
import os

weight = 'bert-base-uncased'
batch_size = 1

abs_path = "./models/" + weight.replace("/", "_")
relay_file = "_pt_model.json"
relay_params = "_pt_model.params"

model_class = transformers.BertModel
tokenizer_class = transformers.BertTokenizer
model = model_class.from_pretrained(weight, return_dict=False)
model.eval()

input_shape = [batch_size, 128]
input_name = 'input_ids'
input_dtype = 'int64'
A = torch.randint(30000, input_shape)
scripted_model = torch.jit.trace(model, [A], strict=False)
shape_list = [('input_ids', input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)


with open(abs_path + relay_file, "w") as fo:
    fo.write(tvm.ir.save_json(mod))
with open(abs_path + relay_params, "wb") as fo:
    fo.write(relay.save_param_dict(params))
