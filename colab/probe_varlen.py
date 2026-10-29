import torch, math, traceback
torch.manual_seed(0)
dev = "cuda"
print("cap", torch.cuda.get_device_capability(), "torch", torch.__version__)
H, D = 4, 256
lq, lk = [5, 7], [9, 7]  # q shorter than k for seq 0 -> bottom-right causal alignment (as in tree packing)
cu_q = torch.tensor([0, 5, 12], dtype=torch.int32, device=dev)
cu_k = torch.tensor([0, 9, 16], dtype=torch.int32, device=dev)
q = torch.randn(12, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(16, H, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(16, H, D, device=dev, dtype=torch.bfloat16)
scale = 1 / math.sqrt(D)


def ref():
    outs = []
    for i in range(2):
        qi = q[cu_q[i]:cu_q[i + 1]].float().transpose(0, 1)
        ki = k[cu_k[i]:cu_k[i + 1]].float().transpose(0, 1)
        vi = v[cu_k[i]:cu_k[i + 1]].float().transpose(0, 1)
        Lq, Lk = qi.shape[1], ki.shape[1]
        mask = torch.ones(Lq, Lk, device=dev, dtype=torch.bool).tril(diagonal=Lk - Lq)
        s = (qi @ ki.transpose(-1, -2)) * scale
        s = s.masked_fill(~mask, float("-inf"))
        outs.append((s.softmax(-1) @ vi).transpose(0, 1))
    return torch.cat(outs)


R = ref()
try:
    from kernels import get_kernel
    fa2 = get_kernel("kernels-community/flash-attn2", version=4)
    o = fa2.flash_attn_varlen_func(q, k, v, cu_q, cu_k, 5, 9, softmax_scale=scale, causal=True, window_size=(-1, -1))
    o = o[0] if isinstance(o, tuple) else o
    print("FA2 hub max err", (o.float() - R).abs().max().item())
    # backward check
    qq, kk, vv = (x.clone().requires_grad_() for x in (q, k, v))
    o = fa2.flash_attn_varlen_func(qq, kk, vv, cu_q, cu_k, 5, 9, softmax_scale=scale, causal=True, window_size=(-1, -1))
    (o[0] if isinstance(o, tuple) else o).float().sum().backward()
    print("FA2 hub backward ok", qq.grad.abs().sum().item() > 0)
except Exception:
    traceback.print_exc()
try:
    from torch.nn.attention.varlen import varlen_attn
    o = varlen_attn(q, k, v, cu_q, cu_k, 5, 9, scale=scale, window_size=(-1, 0))
    print("torch varlen max err", (o.float() - R).abs().max().item())
except Exception:
    traceback.print_exc()
import inspect
from torch.nn.attention.varlen import varlen_attn
print(inspect.signature(varlen_attn))
