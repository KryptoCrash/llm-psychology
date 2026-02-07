from IPython.display import display
from math import sqrt
from einops.einops import einsum
import torch
import transformer_lens
from circuitsvis import attention

model : transformer_lens.HookedTransformer = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""

str_tokens = model.to_str_tokens(model_description_text)
logits, cache = model.run_with_cache(model_description_text, remove_batch_dim = True)

l0_pattern = cache["pattern", 0]
l0_q = cache["q", 0]
l0_k = cache["k", 0]

num_tok, num_heads, dim_head = l0_q.shape
res = einsum(l0_q, l0_k, "tQ h d, tK h d -> h tQ tK")
scaled = res / dim_head ** 0.5
mask = torch.tril(torch.ones_like(scaled)) == 1
masked = torch.where(mask, scaled, float("-inf"))
print(masked)
l0_pattern_from_qk = masked.softmax(-1)

print(l0_pattern.shape, l0_q.shape, l0_k.shape)
print((l0_pattern_from_qk - l0_pattern).abs().max())

html = attention.attention_heads(
    attention=l0_pattern,
    tokens=str_tokens,
    attention_head_names=[f"L0H{i}" for i in range(num_heads)],
)
with open("attn_heads.html", "w") as f: f.write(str(html))
