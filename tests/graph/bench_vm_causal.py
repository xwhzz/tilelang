"""Measure impact of rewriting SDPA attn_mask → is_causal=True on LLaMA-2 prefill."""

import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings; warnings.filterwarnings("ignore")
import operator

import torch
import torch._dynamo
from torch import fx
import tilelang  # noqa: F401
from tilelang.graph import backend_config
from tilelang.graph.cache import clear_cache
from tilelang.profiler import do_bench


# ---------------------------------------------------------------------------
# FX graph pass: rewrite SDPA(attn_mask=causal, is_causal=False)
#             → SDPA(is_causal=True)
# ---------------------------------------------------------------------------

def _is_causal_mask(node: fx.Node) -> bool:
    """Detect if a node produces a standard causal mask (kv_pos <= q_pos).

    Pattern: expand(le(getitem(arange+0, ...), getitem(arange+0, ...)), ...)
    """
    # Walk through expand / reshape
    while node.op == "call_method" and node.target in ("expand", "view", "reshape"):
        node = node.args[0]

    # Should be a le / less_equal comparison
    if node.op != "call_function":
        return False
    if node.target not in (operator.le, torch.le):
        return False

    # Both operands should trace back to arange via getitem/add
    def traces_to_arange(n, depth=0):
        if depth > 5:
            return False
        if not isinstance(n, fx.Node):
            return False
        if n.op == "call_function":
            tgt = n.target
            # arange
            if tgt is torch.arange or (hasattr(tgt, '__name__') and tgt.__name__ == 'arange'):
                return True
            # add(arange, 0) or getitem(arange, slice)
            if tgt in (operator.add, operator.getitem):
                return any(traces_to_arange(a, depth + 1) for a in n.args if isinstance(a, fx.Node))
        return False

    return (len(node.args) >= 2
            and traces_to_arange(node.args[0])
            and traces_to_arange(node.args[1]))


def rewrite_sdpa_causal(gm: fx.GraphModule) -> fx.GraphModule:
    """Rewrite SDPA calls that use an explicit causal mask to use is_causal=True."""
    graph = gm.graph
    modified = False

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is not torch.nn.functional.scaled_dot_product_attention:
            continue

        attn_mask = node.kwargs.get("attn_mask")
        is_causal = node.kwargs.get("is_causal", False)

        if attn_mask is None or is_causal:
            continue

        # Check if attn_mask is a causal mask
        if not isinstance(attn_mask, fx.Node):
            continue
        if not _is_causal_mask(attn_mask):
            continue

        # Rewrite: drop attn_mask, set is_causal=True
        new_kwargs = dict(node.kwargs)
        new_kwargs.pop("attn_mask")
        new_kwargs["is_causal"] = True
        node.kwargs = new_kwargs
        modified = True

    if modified:
        graph.eliminate_dead_code()
        gm.recompile()
        print(f"  Rewrote {sum(1 for n in graph.nodes if n.op == 'call_function' and n.target is torch.nn.functional.scaled_dot_product_attention)} SDPA calls to is_causal=True")

    return gm


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def main():
    from transformers import LlamaForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Model: {model_name}")
    print(f"GPU:   {torch.cuda.get_device_name()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).cuda().eval()

    ids = tokenizer(
        " ".join(["word"] * 1024), return_tensors="pt",
        max_length=512, truncation=True,
    ).input_ids.to("cuda", dtype=torch.int32)
    seq = ids.shape[1]
    print(f"seq_len={seq}\n")

    with torch.no_grad():
        ref = model(ids)

    # --- Inductor baseline (same causal rewrite for fairness) ---
    def inductor_causal_backend(gm, example_inputs):
        gm = rewrite_sdpa_causal(gm)
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(gm, example_inputs)

    torch._dynamo.reset()
    ind = torch.compile(model, backend=inductor_causal_backend)
    with torch.no_grad():
        ind(ids)
        ind_ms = do_bench(lambda: ind(ids))
    print(f"Inductor:              {ind_ms:.2f} ms")

    # --- VM without rewrite ---
    clear_cache(); torch._dynamo.reset()
    backend_config.use_vm = True
    backend_config.vm_clone_output = False
    vm1 = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        out1 = vm1(ids)
        err1 = (out1.logits - ref.logits).abs().max().item()
        vm1_ms = do_bench(lambda: vm1(ids))
    backend_config.use_vm = False
    print(f"VM (no rewrite):       {vm1_ms:.2f} ms  (err={err1:.4f})")

    # --- VM with causal rewrite ---
    from tilelang.graph.backend import tilelang_backend
    def tilelang_causal_backend(gm, example_inputs):
        gm = rewrite_sdpa_causal(gm)
        return tilelang_backend(gm, example_inputs)

    clear_cache(); torch._dynamo.reset()
    backend_config.use_vm = True
    backend_config.vm_clone_output = False
    vm2 = torch.compile(model, backend=tilelang_causal_backend)
    with torch.no_grad():
        out2 = vm2(ids)
        err2 = (out2.logits - ref.logits).abs().max().item()
        vm2_ms = do_bench(lambda: vm2(ids))
    backend_config.use_vm = False
    print(f"VM (is_causal=True):   {vm2_ms:.2f} ms  (err={err2:.4f})")

    print(f"\nGap vs Inductor:  no-rewrite={vm1_ms-ind_ms:+.2f}ms  causal={vm2_ms-ind_ms:+.2f}ms")
    print(f"Saving from rewrite: {vm1_ms - vm2_ms:.2f} ms")

    # Profile the causal rewrite version
    backend_config.use_vm = True
    with torch.no_grad():
        for _ in range(3): vm2(ids)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.no_grad(): vm2(ids)
    torch.cuda.synchronize()
    backend_config.use_vm = False

    print("\n=== VM (causal) top 15 CUDA kernels ===")
    for i, evt in enumerate(sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)):
        if i >= 15: break
        t = evt.self_device_time_total / 1000
        print(f"  {evt.key:<65s} n={evt.count:>4d}  {t:>7.3f}ms")


if __name__ == "__main__":
    main()
