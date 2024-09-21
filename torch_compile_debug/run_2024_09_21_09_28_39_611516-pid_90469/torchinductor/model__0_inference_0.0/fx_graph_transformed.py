class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[100, 100]", arg1_1: "f32[100, 100]"):
        # File: /home/dwarez/Documents/workspace/torch_compile/inductor.py:6 in simple_fn, code: z = torch.matmul(x, y)
        mm: "f32[100, 100]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        
        # File: /home/dwarez/Documents/workspace/torch_compile/inductor.py:7 in simple_fn, code: return torch.nn.functional.softmax(z, dim=1)
        amax: "f32[100, 1]" = torch.ops.aten.amax.default(mm, [1], True)
        sub: "f32[100, 100]" = torch.ops.aten.sub.Tensor(mm, amax);  mm = amax = None
        exp: "f32[100, 100]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[100, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[100, 100]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        return (div,)
        