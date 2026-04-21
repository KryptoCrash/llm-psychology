import os

import plotly.io as pio

import circuitsvis as cv

# Import stuff
import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px

from jaxtyping import Float
from functools import partial

# import transformer_lens
import transformer_lens.utilities as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import FactoredMatrix, HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

torch.set_grad_enabled(False)
device = utils.get_device()

# NBVAL_IGNORE_OUTPUT
model = TransformerBridge.boot_transformers(
    "meta-llama/Llama-3.2-3B-Instruct",
    device=device,
)
model.enable_compatibility_mode(disable_warnings=True)

model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""
loss = model(model_description_text, return_type="loss")
print("Model loss:", loss)

prompt = "The quick brown fox"
output = model.generate(prompt, max_new_tokens=30, do_sample=False)
print("Prompt:", prompt)
print("Output:", output)
