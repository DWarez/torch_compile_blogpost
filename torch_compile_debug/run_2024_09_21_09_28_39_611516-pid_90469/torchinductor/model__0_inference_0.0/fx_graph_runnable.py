
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config


torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.4.1+cu121
# torch cuda version: 12.1
# torch git version: 38b96d3399a695e704ed39b60dac733c3fbf20e2


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Wed_Aug_14_10:10:22_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.68 
# Build cuda_12.6.r12.6/compiler.34714021_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 4060 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        mm = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        amax = torch.ops.aten.amax.default(mm, [1], True)
        sub = torch.ops.aten.sub.Tensor(mm, amax);  mm = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        return (div,)
        
def load_args(reader):
    buf0 = reader.storage(None, 40000)
    reader.tensor(buf0, (100, 100), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 40000)
    reader.tensor(buf1, (100, 100), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)