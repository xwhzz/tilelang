"""End-to-end test: LLaMA-2-7B text generation with TileLang backend."""

import torch
import torch._dynamo
import tilelang  # noqa: F401
from tilelang.graph.cache import clear_cache


def test_llama2_7b_generate():
    from transformers import LlamaForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).cuda().eval()

    prompt = "The capital"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        "cuda", dtype=torch.int32)

    clear_cache()
    torch._dynamo.reset()

    with torch.no_grad():
        out_eager = model.generate(input_ids, max_new_tokens=1, do_sample=False)

        compiled_model = torch.compile(model, backend="tilelang")
        out_tl = compiled_model.generate(input_ids, max_new_tokens=1, do_sample=False)

    text_eager = tokenizer.decode(out_eager[0], skip_special_tokens=True)
    text_tl = tokenizer.decode(out_tl[0], skip_special_tokens=True)

    print(f"Eager:    {text_eager}")
    print(f"TileLang: {text_tl}")

    assert torch.equal(out_eager, out_tl), (
        f"Token mismatch:\n  eager:    {out_eager[0].tolist()}\n"
        f"  tilelang: {out_tl[0].tolist()}")


if __name__ == "__main__":
    test_llama2_7b_generate()
    print("PASS: test_llama2_7b_generate")
