from contextlib import contextmanager
import torch


@contextmanager
def add_hooks(module_forward_hooks=()):
    handles = []
    try:
        for module, hook in module_forward_hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def make_resid_add_hook(vec: torch.Tensor, alpha: float):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * vec.to(h.dtype).to(h.device)
            return (h,) + output[1:]
        return output + alpha * vec.to(output.dtype).to(output.device)
    return hook


@torch.no_grad()
def steer(model, tokenizer, layer: int, vector: torch.Tensor, alpha: float,
          prompts: list[str], max_new_tokens: int = 64, batch_size: int = 8):
    """Add alpha * vector to the residual stream after model.model.layers[layer]
    during inference on prompts. Returns decoded completions."""
    target_module = model.model.layers[layer]
    fwd_hooks = [(target_module, make_resid_add_hook(vector, alpha))]

    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with add_hooks(module_forward_hooks=fwd_hooks):
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = gen[:, enc.input_ids.shape[1]:]
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs
