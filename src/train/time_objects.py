import time
import torch
from torch import nn

from datasets.common import unpack_batch

def time_function(f, reps=100, warmup_reps=100):
    for _ in range(warmup_reps):
        _ = f()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(reps):
        _ = f()
    end.record()
    torch.cuda.synchronize()
    time_per_rep = start.elapsed_time(end) / reps
    return time_per_rep

@torch.no_grad()
def time_model_forward_pass(model, input_shape=None, batch_size=1, device='cpu', **kwargs):
    if hasattr(model, 'input_shape'):
        assert input_shape is None
        input_shape = model.input_shape
    else:
        assert input_shape is not None
    get_input = lambda: torch.randn(batch_size, *input_shape, device=device, dtype=torch.float)
    forward_pass = lambda: model(get_input())
    get_input_time = time_function(get_input, **kwargs)
    combined_time = time_function(forward_pass, **kwargs)
    forward_pass_time = combined_time - get_input_time
    return forward_pass_time

def time_model_backward_pass(model, input_shape=None, batch_size=1, device='cpu', **kwargs):
    if hasattr(model, 'input_shape'):
        assert input_shape is None
        input_shape = model.input_shape
    else:
        assert input_shape is not None
    def backward_pass(do_backward_pass):
        x = torch.randn(batch_size, *input_shape, device=device, dtype=torch.float)
        y = torch.randint(256, (batch_size,), device=device, dtype=torch.long)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        model.zero_grad(set_to_none=True)
        if do_backward_pass:
            loss.backward()
    baseline_time = time_function(lambda: backward_pass(False), **kwargs)
    combined_time = time_function(lambda: backward_pass(True), **kwargs)
    backward_pass_time = combined_time - baseline_time
    return backward_pass_time

def time_dataloader(dataloader):
    def iterate_through_dataloader():
        for _ in dataloader:
            pass
    dataloader_time = time_function(iterate_through_dataloader, reps=1, warmup_reps=0)
    return dataloader_time
