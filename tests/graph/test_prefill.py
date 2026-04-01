"""Correctness test: LLaMA-2-7B prefill with TileLang backend."""

import torch
import torch._dynamo
import tilelang  # noqa: F401
from tilelang.graph.cache import clear_cache


def test_llama2_7b_prefill():
    from transformers import LlamaForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).cuda().eval()

    ids = tokenizer(
        " ".join(["word"] * 1024), return_tensors="pt",
        max_length=512, truncation=True,
    ).input_ids.to("cuda", dtype=torch.int32)

    clear_cache()
    torch._dynamo.reset()

    with torch.no_grad():
        ref = model(ids)
        compiled = torch.compile(model, backend="tilelang")
        out = compiled(ids)

    max_err = (out.logits - ref.logits).abs().max().item()
    print(f"Prefill seq={ids.shape[1]}: max_err={max_err:.4f}")
    assert max_err < 0.2, f"Prefill error too large: {max_err}"


if __name__ == "__main__":
    test_llama2_7b_prefill()
    print("PASS: test_llama2_7b_prefill")
