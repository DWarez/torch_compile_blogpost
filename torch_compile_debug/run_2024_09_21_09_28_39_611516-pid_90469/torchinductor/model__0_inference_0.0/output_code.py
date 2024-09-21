
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


cpp_fused__softmax_0 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_dwarez/sk/cskh5dx62fglpphcrl6723dnmowdabouerrzy3dmqcngbxwfa7bv.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100L*x0)), 8);
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(96L); x1<static_cast<long>(100L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (100L*x0))];
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100L*x0)), 8);
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(out_ptr1 + static_cast<long>(x1 + (100L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(96L); x1<static_cast<long>(100L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (100L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp3 = std::exp(tmp2);
                    out_ptr1[static_cast<long>(x1 + (100L*x0))] = tmp3;
                    tmp_acc0 = tmp_acc0 + tmp3;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (100L*x0)), 8);
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr3 + static_cast<long>(x1 + (100L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(96L); x1<static_cast<long>(100L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr1[static_cast<long>(x1 + (100L*x0))];
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[static_cast<long>(x1 + (100L*x0))] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (100, 100), (100, 1))
    assert_size_stride(arg1_1, (100, 100), (100, 1))
    buf0 = empty_strided_cpu((100, 100), (100, 1), torch.float32)
    # Source Nodes: [z], Original ATen: [aten.mm]
    extern_kernels.mm(arg1_1, arg0_1, out=buf0)
    del arg0_1
    del arg1_1
    buf1 = empty_strided_cpu((100, 1), (1, 100), torch.float32)
    buf2 = empty_strided_cpu((100, 100), (100, 1), torch.float32)
    buf3 = empty_strided_cpu((100, 1), (1, 100), torch.float32)
    buf4 = empty_strided_cpu((100, 100), (100, 1), torch.float32)
    cpp_fused__softmax_0(buf0, buf1, buf2, buf3, buf4)
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((100, 100), (100, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((100, 100), (100, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
